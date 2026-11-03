"""
Strategy: Spherical Lloyd's Algorithm with Simulated Annealing.
1. Compute Voronoi cells on the sphere via Convex Hull facets.
2. Move each point to the centroid of its Voronoi cell (Lloyd's step).
3. Use Simulated Annealing to accept worse moves probabilistically to escape local minima.
4. Exact covering radius computed via Convex Hull circumcenters.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) using the Convex Hull.
    The covering radius is the maximum of the circumradii of the facets
    (Voronoi cells) of the convex hull of the points.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails (e.g., points are coplanar or degenerate)
        # Use a sampling-based estimate
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    # For each facet (triangle in 3D hull), compute the circumradius on the sphere
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Vectors
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
            
        # Circumcenter of the triangle in 3D
        # We need the point on the sphere that is equidistant from a, b, c.
        # This is the intersection of the line through the origin in the direction of the normal
        # (or its opposite) with the sphere.
        # The normal vector n is perpendicular to the plane of the triangle.
        n = cross / norm_cross
        
        # The circumcenter on the sphere is either n or -n.
        # We check which one is inside the Voronoi cell (or simply check the distance).
        # Actually, for a convex hull of points on the sphere, the facet normal points outward.
        # The "empty cap" center is the point on the sphere furthest from the origin in the direction
        # of the normal, provided it's "visible" from the outside.
        # However, the distance from the facet vertices to the normal direction is the circumradius.
        
        dot_a = np.dot(n, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        # Check the other side just in case (though for convex hull, one side is the interior)
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        # The covering radius is the max distance to the nearest point.
        # For a facet, the furthest point in the cell from the vertices is the circumcenter.
        # The distance is arccos(dot(n, a)).
        # We take the max over all facets.
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
            
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius via sampling."""
    n_theta = 100
    n_phi = 200
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _lloyd_step(pts):
    """
    Perform one iteration of Spherical Lloyd's algorithm.
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts, False
        
    new_pts = np.zeros_like(pts)
    
    # For each point, find the facets (triangles) that contain it in their simplex.
    # The Voronoi cell on the sphere corresponds to the union of the "spherical triangles"
    # formed by projecting the hull facets onto the sphere.
    # A simpler approximation for the centroid:
    # The centroid of the Voronoi cell is proportional to the sum of the normals of the facets
    # adjacent to that point, weighted by the area of the facet.
    
    # Build a map: point index -> list of facet indices
    point_to_facets = {i: [] for i in range(N)}
    for i, simplex in enumerate(hull.simplices):
        for p_idx in simplex:
            point_to_facets[p_idx].append(i)
            
    for i in range(N):
        facets = point_to_facets[i]
        if not facets:
            new_pts[i] = pts[i]
            continue
            
        centroid_vec = np.zeros(3)
        total_weight = 0.0
        
        for f_idx in facets:
            simplex = hull.simplices[f_idx]
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            # Area of the spherical triangle is proportional to the solid angle.
            # For small triangles, it's proportional to the planar area.
            # Let's use the planar area as a weight.
            v1 = b - a
            v2 = c - a
            cross = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross)
            
            # The normal to the facet
            n = cross / np.linalg.norm(cross)
            
            # The "centroid" of the facet on the sphere is roughly n (or -n).
            # For a convex hull, the normal points outward.
            # The Voronoi cell is on the "inside" of the hull relative to the origin?
            # No, the points are on the sphere. The hull is the convex hull of the points.
            # The Voronoi cell of point i is the set of points on the sphere closest to i.
            # This corresponds to the region of the sphere "covered" by the facets adjacent to i.
            # The normal n points away from the origin.
            # We should sum the normals.
            
            centroid_vec += n * area
            total_weight += area
            
        if total_weight > 1e-12:
            new_pts[i] = centroid_vec / total_weight
        else:
            new_pts[i] = pts[i]
            
    new_pts = _to_sphere(new_pts)
    return new_pts, True

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _covering_radius(pts)
    
    deadline = _DEADLINE
    
    # Simulated Annealing parameters
    T = 0.5  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    np.random.seed(42)
    
    while time.time() < deadline - 10:
        # 1. Lloyd's Step (Deterministic improvement)
        new_pts, success = _lloyd_step(pts)
        if success:
            new_obj = _covering_radius(new_pts)
            if new_obj < best_obj:
                best_obj = new_obj
                best_pts = new_pts.copy()
            pts = new_pts
            
        # 2. Metropolis Step (Stochastic escape)
        if T > T_min:
            # Perturb the points
            noise_scale = T * 0.1
            perturbed = _to_sphere(pts + np.random.randn(N, 3) * noise_scale)
            perturbed_obj = _covering_radius(perturbed)
            
            # Accept if better, or with probability exp(-(delta)/T)
            delta = perturbed_obj - pts_obj
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                pts = perturbed
                pts_obj = perturbed_obj
            else:
                # Keep current pts, but update pts_obj if it was stale? 
                # We should maintain pts_obj consistently.
                pass
        
        # Cool down
        T *= cooling_rate
        
        # Update pts_obj if we didn't just evaluate it
        if not success:
             pts_obj = _covering_radius(pts)

    # Final Polish: Run a few pure Lloyd iterations without noise to converge
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        new_pts, success = _lloyd_step(pts)
        if success:
            new_obj = _covering_radius(new_pts)
            if new_obj < best_obj:
                best_obj = new_obj
                best_pts = new_pts.copy()
            pts = new_pts
        else:
            break

    # Ensure best_pts is the absolute best found
    final_obj = _covering_radius(best_pts)
    if final_obj > best_obj:
        # This can happen if best_pts was updated but best_obj wasn't synced perfectly
        best_obj = final_obj

    return best_pts
# EVOLVE-BLOCK-END
