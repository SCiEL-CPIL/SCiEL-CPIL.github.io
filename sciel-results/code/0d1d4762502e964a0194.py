"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees by checking hull facets."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return 180.0
    if not np.all(np.isfinite(pts)):
        return 180.0
    
    # Check for coincident points
    # Efficient check using cdist or simple loop for small N
    # N=130, N^2/2 = 8450 checks, fast enough
    min_dist = np.inf
    for i in range(N):
        dists = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        if len(dists) > 0:
            min_dist = min(min_dist, np.min(dists))
    if min_dist < 1e-12:
        return 180.0
        
    # Check hemisphere condition: origin inside convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # For scipy ConvexHull, equations are n_i · x + d_i = 0.
    # Interior: n_i · x + d_i <= 0.
    # Origin (0,0,0) -> d_i <= 0.
    # If any d_i > 0, origin is outside.
    if np.any(hull.equations[:, -1] > 1e-10):
        return 180.0

    max_ang = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Compute circumcenter of triangle on unit sphere
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
        
        # Check if c is inside the spherical triangle
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_ang) * 180.0 / np.pi
        else:
            mid01 = (v0 + v1) / np.linalg.norm(v0 + v1)
            mid12 = (v1 + v2) / np.linalg.norm(v1 + v2)
            mid20 = (v2 + v0) / np.linalg.norm(v2 + v0)
            
            ang01 = np.arccos(np.clip(np.dot(mid01, v0), -1.0, 1.0)) * 180.0 / np.pi
            ang12 = np.arccos(np.clip(np.dot(mid12, v1), -1.0, 1.0)) * 180.0 / np.pi
            ang20 = np.arccos(np.clip(np.dot(mid20, v2), -1.0, 1.0)) * 180.0 / np.pi
            
            ang = max(ang01, ang12, ang20)
        
        max_ang = max(max_ang, ang)
    
    return max_ang

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_step(pts):
    """
    Perform one Lloyd iteration: move each point to the centroid of its Voronoi cell.
    On the sphere, the Voronoi cell of point p is the set of points y such that y·p >= y·q for all q.
    The convex hull of the points defines the dual polytope.
    The Voronoi cell of point i corresponds to the face of the convex hull opposite to point i?
    No, that's for the polar dual.
    
    Actually, let's use the ConvexHull to find the neighbors.
    The Voronoi cell of point i on the sphere is the convex hull of the circumcenters of all facets
    (triangles) that contain point i, plus the points themselves?
    No, the Voronoi cell is a spherical polygon. Its vertices are the circumcenters of the
    spherical triangles (facets of the convex hull) that include point i.
    
    The centroid of the Voronoi cell can be approximated by the normalized sum of its vertices
    (circumcenters) weighted by the solid angle or just uniformly.
    A common approximation for spherical centroids is the normalized sum of the vertices of the cell.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts

    # Map point index to list of facet indices containing it
    point_to_facets = {i: [] for i in range(N)}
    for idx, simplex in enumerate(hull.simplices):
        for p_idx in simplex:
            point_to_facets[p_idx].append(idx)
    
    new_pts = pts.copy()
    for i in range(N):
        facets = point_to_facets[i]
        if not facets:
            continue
            
        # Compute circumcenters of these facets
        centroids = []
        for f_idx in facets:
            simplex = hull.simplices[f_idx]
            v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                c = (v0 + v1 + v2) / 3.0
                c = c / np.linalg.norm(c)
            else:
                c = n / n_norm
            centroids.append(c)
            
        # The Voronoi cell is the convex hull of these circumcenters.
        # Approximate centroid: normalized sum of circumcenters.
        # To be more accurate, we could compute the centroid of the spherical polygon,
        # but the sum of vertices is a decent first-order approximation.
        # Better: average the circumcenters.
        centroid = np.sum(centroids, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            new_pts[i] = centroid / norm
        else:
            # Keep original if centroid is zero (shouldn't happen)
            pass
            
    return _normalize(new_pts)

def _smooth_max_objective(x, beta=50.0):
    """
    Objective: smooth max of facet circumradii.
    This encourages equalizing the worst facets.
    """
    pts = _normalize(x.reshape(N, 3))
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
        
    # Check origin inside hull
    if np.any(hull.equations[:, -1] > 1e-10):
        return 180.0

    radii = []
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
            
        # Check inside
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_ang)
        else:
            mid01 = (v0 + v1) / np.linalg.norm(v0 + v1)
            mid12 = (v1 + v2) / np.linalg.norm(v1 + v2)
            mid20 = (v2 + v0) / np.linalg.norm(v2 + v0)
            
            ang01 = np.arccos(np.clip(np.dot(mid01, v0), -1.0, 1.0))
            ang12 = np.arccos(np.clip(np.dot(mid12, v1), -1.0, 1.0))
            ang20 = np.arccos(np.clip(np.dot(mid20, v2), -1.0, 1.0))
            ang = max(ang01, ang12, ang20)
            
        radii.append(ang)
        
    radii = np.array(radii)
    # Smooth max: (1/beta) * log(sum(exp(beta * r)))
    # Use log-sum-exp trick for stability
    max_r = np.max(radii)
    log_sum_exp = max_r + np.log(np.sum(np.exp(beta * (radii - max_r))))
    return log_sum_exp / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = _fibonacci_spiral()
    best_score = _covering_radius_deg(best_pts)
    
    # Multi-restart Lloyd's Algorithm
    np.random.seed(42)
    
    # Generate a few initial configurations
    inits = [best_pts]
    for _ in range(5):
        # Random jittered Fibonacci
        jitter = np.random.randn(N, 3) * 0.2
        init = _normalize(_fibonacci_spiral() + jitter)
        inits.append(init)
        
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        current_pts = init.copy()
        current_score = _covering_radius_deg(current_pts)
        
        # Lloyd's iterations
        for i in range(100):
            if time.time() > deadline - 20.0:
                break
                
            next_pts = _lloyd_step(current_pts)
            next_score = _covering_radius_deg(next_pts)
            
            if next_score < current_score - 1e-8:
                current_pts = next_pts
                current_score = next_score
            else:
                # Stagnation, break
                break
                
        if current_score < best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
    # Fine-tuning with smooth-max gradient if time permits
    if time.time() < deadline - 10.0:
        x0 = best_pts.flatten()
        # We don't have an easy gradient for smooth_max_objective due to hull discontinuities.
        # Use Nelder-Mead or L-BFGS-B with numerical gradient?
        # L-BFGS-B with numerical gradient is slow.
        # Let's try a few random perturbations + Lloyd again.
        
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
                
            noise = np.random.randn(N, 3) * 0.05
            perturbed = _normalize(best_pts + noise)
            
            # Run Lloyd on perturbed
            curr = perturbed
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                nxt = _lloyd_step(curr)
                s = _covering_radius_deg(nxt)
                if s < _covering_radius_deg(curr) - 1e-8:
                    curr = nxt
                else:
                    break
                    
            s = _covering_radius_deg(curr)
            if s < best_score:
                best_score = s
                best_pts = curr.copy()
                
    return best_pts
# EVOLVE-BLOCK-END
