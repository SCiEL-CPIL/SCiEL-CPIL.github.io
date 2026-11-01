import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
_DEADLINE_OFFSET = 10.0  # Safety margin

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_data(pts):
    """
    Compute convex hull and extract facet data.
    Returns:
        radii: (F,) array of circumradii (angular distance from vertex to circumcenter)
        normals: (F, 3) array of outward unit normals (pointing towards circumcenter)
        faces: (F, 3) array of vertex indices
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute circumcenter for each triangle on the unit sphere
        # The circumcenter is the intersection of the plane ABC with the line through origin perpendicular to the plane.
        # For a triangle with vertices A, B, C on the unit sphere, the circumcenter direction is proportional to 
        # (A x B) x C + ... or simpler: normalize the vector sum of vertices weighted by opposite side lengths?
        # Actually, the circumcenter of a spherical triangle is the pole of the great circle passing through A, B, C.
        # The normal to the plane containing A, B, C is n = (B-A) x (C-A).
        # The circumcenter direction is n / ||n|| if the triangle is "small" enough (less than hemisphere).
        # Wait, the circumcenter of the triangle on the sphere is the point P such that P.A = P.B = P.C = cos(R).
        # P is along the normal n. P = n / ||n|| or -n / ||n||.
        # We want the one that is "outside" the convex hull? No, the covering radius is the max distance to the nearest point.
        # The circumcenter of a facet (triangle) is the point furthest from the vertices *within that facet's cap*.
        # The normal vector n points outward from the hull. The circumcenter is in the direction of n.
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        valid = norms > _EPS
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([])
        
        normals = cross[valid]
        normals = normals / (norms + _EPS)
        
        # Check orientation: normals should point away from the origin (outward)
        # Since it's a convex hull of points on the sphere, the centroid of the face should be in the direction of the normal.
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The angular radius of the circumcircle is arccos(dot(normal, A))
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid]
    except Exception:
        return np.array([]), np.array([]), np.array([])

def _covering_radius(pts):
    """Compute the covering radius."""
    radii, _, _ = _get_hull_data(pts)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def _lloyd_step(pts):
    """
    Perform one Lloyd iteration: move each point towards the circumcenter of its facet.
    """
    radii, normals, faces = _get_hull_data(pts)
    if len(radii) == 0:
        return pts
    
    new_pts = pts.copy()
    
    # For each vertex, find the facets it belongs to
    # We want to move the vertex towards the circumcenter of the "worst" facet it belongs to,
    # or a weighted average. Standard Lloyd for covering is to move towards the circumcenter of the 
    # Voronoi cell. For points on the sphere, the Voronoi cell of a vertex is the region of the sphere 
    # closer to that vertex than any other. The "center" of this cell is not simply the facet circumcenter.
    # However, a common heuristic is to move the vertex towards the pole of the facet with the largest radius 
    # that it is part of, or to average the normals of the adjacent facets.
    
    # Let's use a simpler, robust approach:
    # Identify the worst facet (largest radius).
    # Move its 3 vertices towards the pole of that facet.
    # This directly reduces the largest radius.
    
    worst_idx = np.argmax(radii)
    worst_face = faces[worst_idx]
    pole = normals[worst_idx]
    
    # Move vertices of the worst face towards the pole
    for v in worst_face:
        # Direction from vertex to pole
        dir_vec = pole - pts[v]
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > _EPS:
            # Move a fraction of the way
            # Step size needs to be controlled to avoid overshooting
            step = 0.1 
            new_pts[v] = pts[v] + step * dir_vec
            new_pts[v] /= (np.linalg.norm(new_pts[v]) + _EPS)
            
    return new_pts

def _local_search(pts, deadline):
    """
    Iterative local search reducing the maximum facet radius.
    """
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    while time.time() < deadline:
        radii, normals, faces = _get_hull_data(pts)
        if len(radii) == 0:
            break
        
        curr_r = np.max(radii)
        if curr_r < best_r - 1e-10:
            best_r = curr_r
            best_pts = pts.copy()
        
        # If we've reached a very good solution, stop
        if best_r < np.deg2rad(11.32):
            break
            
        # Find the worst facet
        worst_idx = np.argmax(radii)
        worst_face = faces[worst_idx]
        pole = normals[worst_idx]
        
        # Try to improve by moving the vertices of the worst facet
        # We can try different step sizes
        improved = False
        for step in [0.5, 0.2, 0.1, 0.05, 0.01]:
            cand = pts.copy()
            for v in worst_face:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    cand[v] = pts[v] + step * dir_vec
                    cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
            
            cand_r = _covering_radius(cand)
            if cand_r < curr_r - 1e-10:
                pts = cand
                improved = True
                break
        
        if not improved:
            # If we can't improve the worst facet by moving its vertices, 
            # maybe we should try a small random perturbation of one of the vertices?
            # Or break to avoid infinite loop on a local minimum.
            # Let's try a small random kick to the worst vertex
            v = worst_face[np.argmax(np.linalg.norm(pts[worst_face] - pole, axis=1))]
            rng = np.random.default_rng(hash(v) % 2**32)
            kick = rng.standard_normal(3) * 0.01
            cand = pts.copy()
            cand[v] = pts[v] + kick
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
            
            if _covering_radius(cand) < curr_r:
                pts = cand
            else:
                break
                
    return best_pts, best_r

def _annealing(pts, deadline):
    """
    Simulated annealing to escape local minima.
    """
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    curr_pts = pts.copy()
    curr_r = best_r
    
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.95
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > T_min:
        # Perturb a random point
        idx = rng.integers(0, N)
        cand = curr_pts.copy()
        
        # Random direction on sphere
        dir = rng.standard_normal(3)
        dir /= (np.linalg.norm(dir) + _EPS)
        
        # Step size proportional to T
        step = T * 0.1
        cand[idx] = curr_pts[idx] + step * dir
        cand[idx] /= (np.linalg.norm(cand[idx]) + _EPS)
        
        cand_r = _covering_radius(cand)
        
        delta = cand_r - curr_r
        if delta < 0 or rng.random() < np.exp(-delta / T):
            curr_pts = cand
            curr_r = cand_r
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = curr_pts.copy()
        
        T *= cooling_rate
        
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    pts = _initial_fib()
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    # Phase 1: Local Search
    # Run multiple local searches from slightly perturbed starts
    n_restarts = 3
    time_per_restart = (deadline - 20.0) / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if i == 0:
            curr_pts = best_pts.copy()
        else:
            # Perturb best solution
            rng = np.random.default_rng(100 + i)
            curr_pts = best_pts + rng.standard_normal((N, 3)) * 0.1
            curr_pts /= np.linalg.norm(curr_pts, axis=1, keepdims=True)
        
        # Run local search
        new_pts, new_r = _local_search(curr_pts, deadline - 20.0 - (n_restarts - i - 1) * time_per_restart)
        
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts
    
    # Phase 2: Annealing from the best solution
    if time.time() < deadline - 5.0:
        new_pts, new_r = _annealing(best_pts, deadline - 5.0)
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts

    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts /= (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
