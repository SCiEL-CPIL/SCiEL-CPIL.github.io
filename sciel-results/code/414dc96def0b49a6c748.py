"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Constants
_DEADLINE = 142.0  # Safety margin before 150s kill
_N_SAMPLES = 50000  # Number of sample points for K-Means

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _generate_random_samples(n_samples, rng):
    """Generate random points on the sphere."""
    # Use normal distribution and normalize
    pts = rng.normal(size=(n_samples, 3))
    return _normalize(pts)

def _spherical_kmeans(samples, k, rng, max_iters=100):
    """
    Perform Spherical K-Means clustering.
    Returns centroids (k, 3).
    """
    n_samples = len(samples)
    
    # Initialize centroids randomly from samples
    init_indices = rng.choice(n_samples, size=k, replace=False)
    centroids = samples[init_indices].copy()
    
    for _ in range(max_iters):
        # Assignment step: find nearest centroid for each sample
        # dot product is proportional to cos(theta)
        dots = samples @ centroids.T
        nearest_idx = dots.argmax(axis=1)
        
        # Update step: compute spherical mean for each cluster
        new_centroids = np.zeros((k, 3))
        for j in range(k):
            mask = (nearest_idx == j)
            if np.sum(mask) > 0:
                cluster_points = samples[mask]
                # Spherical mean is the normalized vector sum
                mean_vec = np.sum(cluster_points, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-10:
                    new_centroids[j] = mean_vec / norm
                else:
                    # If points are evenly distributed on the sphere, mean is ~0
                    # Keep old centroid or pick a random point from cluster
                    new_centroids[j] = cluster_points[0]
            else:
                # If a cluster is empty, re-seed with a random sample
                new_centroids[j] = samples[rng.integers(0, n_samples)]
        
        # Check convergence
        diff = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        if diff < 1e-6:
            break
            
    return centroids

def _covering_radius_exact(pts):
    """
    Calculate the exact covering radius using ConvexHull.
    The covering radius is the maximum circumradius of the facets of the convex hull.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            # Each simplex is a triangle (in 3D)
            # We need the circumradius of the spherical triangle
            # The circumradius R of a spherical triangle with side lengths a, b, c 
            # can be found, but a simpler method for the "cap" radius is to find 
            # the point equidistant from the 3 vertices (spherical circumcenter).
            
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # The circumcenter of the spherical triangle is the normalized 
            # linear combination of the vertices.
            # Alternatively, we can find the point on the sphere that maximizes 
            # the minimum distance to the 3 vertices? No, the circumcenter is the 
            # center of the circumscribed circle.
            
            # A robust way: Solve for the circumcenter on the sphere.
            # The circumcenter C is in the plane defined by v1, v2, v3? No.
            # The circumcenter is the intersection of the sphere with the line 
            # passing through the origin and the centroid of the triangle? No.
            
            # Let's use the property that the circumcenter C satisfies:
            # |C - v1|^2 = |C - v2|^2 = |C - v3|^2 = 2(1 - cos(R))
            # Since |C|=1, |v|=1:
            # 1 + 1 - 2 C.v1 = 1 + 1 - 2 C.v2 => C.v1 = C.v2
            # So C is in the plane perpendicular to v1-v2 and v1-v3 passing through 0?
            # Actually, C is in the span of v1, v2, v3.
            
            # Let C = a*v1 + b*v2 + c*v3
            # This is a linear system.
            # However, a simpler numerical approach:
            # The circumradius is the distance from the circumcenter to any vertex.
            # We can find the circumcenter by finding the intersection of the 
            # great circle bisectors.
            
            # Efficient method: 
            # The circumcenter is the normalized cross product of the normals to the 
            # great circles bisecting the sides?
            
            # Let's just use a small local optimization to find the point on the 
            # sphere that is equidistant to v1, v2, v3.
            # Or better: The circumradius of the spherical triangle is given by 
            # the formula involving the area and side lengths, but that's complex.
            
            # Alternative: The covering radius is the max over all points y on sphere 
            # of min_i d(y, p_i). The maximum of this function occurs at the 
            # circumcenters of the Voronoi cells (facets).
            
            # For a triangle v1, v2, v3, the circumcenter C can be found by solving:
            # C . (v1 - v2) = 0
            # C . (v1 - v3) = 0
            # |C| = 1
            # This implies C is perpendicular to the vectors (v1-v2) and (v1-v3).
            # So C is parallel to (v1-v2) x (v1-v3).
            
            normal1 = np.cross(v1 - v2, v1 - v3)
            if np.linalg.norm(normal1) < 1e-10:
                # Degenerate triangle (collinear points on sphere)
                # This shouldn't happen for a convex hull facet
                continue
            c = normal1 / np.linalg.norm(normal1)
            
            # There are two possible circumcenters (c and -c).
            # The correct one is the one that is "inside" the Voronoi cell, 
            # which means it should have a larger dot product with the vertices 
            # than the opposite direction? No, the circumcenter is the center of the 
            # circle passing through the vertices.
            # The distance from C to V1 is arccos(C.V1).
            # We want the radius R = arccos(C.V1).
            # We should pick the C that gives the smaller radius? 
            # Actually, the Voronoi vertex is the point that is equidistant to v1, v2, v3 
            # and is the "center" of the empty cap.
            # The dot product C.V1 will be the same for C and -C? No.
            # (-C).V1 = -C.V1.
            # One will give a small angle, one a large angle.
            # The circumradius of the facet is the angular distance from the circumcenter 
            # to the vertices. We want the one that corresponds to the "empty" cap, 
            # which is the one with the larger distance? No, the covering radius is the 
            # distance to the *nearest* point. The worst-case point is the circumcenter 
            # of the facet.
            
            # Let's calculate the dot product for c.
            dot1 = np.dot(c, v1)
            dot2 = np.dot(-c, v1)
            
            # The circumcenter of the spherical triangle is the one that lies in the 
            # convex hull of the vertices? No.
            # The Voronoi vertex is the point that maximizes the minimum distance to 
            # the set of points. For the specific facet, it's the circumcenter.
            # There are two points on the sphere equidistant from v1, v2, v3.
            # One is the circumcenter of the triangle, the other is the antipode.
            # The circumcenter of the triangle (the one "inside") usually has a 
            # larger dot product with the vertices if the triangle is small?
            
            # Let's check the distance.
            r1 = np.arccos(np.clip(dot1, -1, 1))
            r2 = np.arccos(np.clip(dot2, -1, 1))
            
            # The radius of the cap is the distance to the nearest point.
            # If we pick the wrong circumcenter, the distance to v1, v2, v3 will be 
            # large, but there might be other points closer.
            # However, the covering radius is the MAX over the sphere.
            # The local maxima of the distance function are the Voronoi vertices.
            # The value of the function at a Voronoi vertex is the distance to its 
            # defining points.
            
            # We should take the minimum of the two radii? 
            # No, the Voronoi vertex is unique. 
            # The correct circumcenter is the one that is in the same hemisphere 
            # as the centroid of the triangle?
            
            centroid = (v1 + v2 + v3) / 3
            if np.dot(c, centroid) < 0:
                c = -c
                dot1 = -dot1
            
            # Now c is the circumcenter.
            # The distance from c to v1 is the circumradius.
            radius = np.degrees(np.arccos(np.clip(dot1, -1, 1)))
            
            if radius > max_circumradius:
                max_circumradius = radius
                
        return max_circumradius
    except Exception:
        # Fallback to sampling if hull fails
        samples = _generate_random_samples(10000, np.random.default_rng(0))
        dots = samples @ pts.T
        max_dots = dots.max(axis=1)
        min_max_dot = np.min(max_dots)
        return np.degrees(np.arccos(np.clip(min_max_dot, -1, 1)))

def _local_optimize(pts, samples, max_iters=2000, time_limit=None, start_time=None):
    """
    Local optimization using gradient-like steps based on farthest assigned samples.
    """
    if start_time is None:
        start_time = time.time()
        
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_radius = _covering_radius_exact(pts)
    
    lr = 0.02  # Learning rate
    
    for it in range(max_iters):
        if time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        # Compute nearest configuration point for each sample
        dots = samples @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        nearest_idx = dots.argmax(axis=1)  # (M,)
        
        new_pts = pts.copy()
        
        # For each configuration point, find the farthest assigned sample
        for j in range(N):
            mask = (nearest_idx == j)
            if not np.any(mask):
                continue
            
            assigned_samples = samples[mask]
            dots_j = dots[mask, j]
            
            # Find the sample with minimum dot product (farthest)
            far_idx = np.argmin(dots_j)
            far_sample = assigned_samples[far_idx]
            
            # Move point j towards far_sample
            diff = far_sample - pts[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                new_pts[j] = pts[j] + lr * diff / diff_norm
        
        new_pts = _normalize(new_pts)
        
        # Evaluate
        new_radius = _covering_radius_exact(new_pts)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            lr = max(0.005, lr * 0.999)
        else:
            lr = max(0.003, lr * 0.95)
    
    return best_pts, best_radius

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses Spherical K-Means for initialization and local optimization for polishing.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Generate fixed samples for K-Means and evaluation
    rng = np.random.default_rng(42)
    samples = _generate_random_samples(_N_SAMPLES, rng)
    
    # Track best result
    best_pts = None
    best_radius = np.inf
    
    # Strategy: Multiple Spherical K-Means restarts
    n_restarts = 10
    
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # 1. Spherical K-Means Initialization
        # Use a different seed for each restart
        restart_rng = np.random.default_rng(42 + i)
        centroids = _spherical_kmeans(samples, N, restart_rng, max_iters=50)
        
        # 2. Local Optimization Polish
        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
            
        opt_pts, opt_radius = _local_optimize(centroids, samples, max_iters=1000, 
                                               time_limit=remaining, 
                                               start_time=start_time)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
            
    if best_pts is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
