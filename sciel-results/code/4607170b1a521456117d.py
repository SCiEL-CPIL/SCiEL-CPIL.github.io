import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and angular radii of all facets.
    Returns:
        max_radius: The maximum angular radius (covering radius)
        worst_idx: Index of the worst facet in the filtered array
        v_indices: Indices of the 3 vertices of the worst facet
        pole: The normal vector (circumcenter direction) of the worst facet
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, None, None, None
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: arccos(normal . vertex)
        # We need the radius for each vertex of each facet to find the max
        # For a facet with normal n and vertices a,b,c, the angular distances are arccos(n.a), etc.
        # The covering radius of the facet is the max of these three.
        
        # Let's compute the max radius for each facet
        # normals shape (M_valid, 3)
        # A[valid] shape (M_valid, 3)
        
        cos_theta_a = np.sum(normals * A[valid], axis=1)
        cos_theta_b = np.sum(normals * B[valid], axis=1)
        cos_theta_c = np.sum(normals * C[valid], axis=1)
        
        # Clip to [-1, 1] for arccos
        cos_theta_a = np.clip(cos_theta_a, -1.0, 1.0)
        cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)
        cos_theta_c = np.clip(cos_theta_c, -1.0, 1.0)
        
        rad_a = np.arccos(cos_theta_a)
        rad_b = np.arccos(cos_theta_b)
        rad_c = np.arccos(cos_theta_c)
        
        facet_radii = np.maximum(np.maximum(rad_a, rad_b), rad_c)
        
        worst_idx = np.argmax(facet_radii)
        max_radius = facet_radii[worst_idx]
        
        # Get the vertices and pole for the worst facet
        # Note: faces[valid] gives the indices for the valid facets
        v_indices = faces[valid][worst_idx]
        pole = normals[worst_idx]
        
        return max_radius, worst_idx, v_indices, pole
    except Exception:
        return np.pi, None, None, None

def _slerp(p, q, t):
    """Spherical linear interpolation."""
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)
    dot = np.dot(p, q)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    
    if omega < _EPS:
        return p
        
    so = np.sin(omega)
    # Handle case where omega is close to pi
    if so < _EPS:
        return (1-t)*p - t*q
        
    s1 = np.sin((1-t)*omega) / so
    s2 = np.sin(t*omega) / so
    return s1 * p + s2 * q

def _try_improve_worst_facet(pts, v_indices, pole, current_max_radius, rng):
    """
    Attempts to reduce the global covering radius by moving the vertices of the worst facet
    towards the pole (circumcenter). Uses a line search along the geodesic.
    """
    if v_indices is None:
        return None, current_max_radius
        
    best_candidate = None
    best_radius = current_max_radius
    
    # We want to move vertices towards the pole.
    # The "worst" vertex is the one with the largest angular distance to the pole.
    # However, moving all three might be necessary or just one.
    # Let's try moving all three simultaneously towards the pole by a fraction t.
    # Or, identify the specific vertex that is the bottleneck for this facet and move it.
    
    # Calculate angular distances of the 3 vertices to the pole
    dists = []
    for v in v_indices:
        d = np.arccos(np.clip(np.dot(pts[v], pole), -1.0, 1.0))
        dists.append(d)
        
    # The facet radius is max(dists).
    # To reduce the facet radius, we should move the vertex with the largest distance.
    # But moving it might increase the radius of adjacent facets.
    # Strategy: Binary search for the maximum step t in [0, 1] such that the global max radius decreases.
    
    # Actually, a simpler and robust strategy:
    # Try to move the "worst" vertex (largest dist) towards the pole.
    # If that fails or increases global max, try moving the next worst, etc.
    # Or, move all three proportionally?
    
    # Let's stick to the "worst vertex" heuristic first, but also consider that
    # the facet radius is determined by the max of the three.
    # If we move the vertex with dist_i towards the pole, dist_i decreases.
    # If dist_i was the max, the facet radius decreases.
    
    # Let's try a line search for moving the single worst vertex first.
    worst_local_idx = np.argmax(dists)
    target_v = v_indices[worst_local_idx]
    
    # Binary search for step size t in (0, 1)
    # t=0 is current state. t=1 is vertex at pole.
    # We want the largest t such that new_global_max < current_max_radius.
    # Actually, we just want ANY improvement. Let's find the best t that minimizes global max.
    
    # Since the function is not strictly convex, we do a coarse-to-fine search.
    # Or simply: try a few candidate t values.
    
    # A more robust approach: 
    # The ideal position for the vertex to minimize the facet radius is the pole itself (if we ignore neighbors).
    # But we are constrained by neighbors.
    
    # Let's do a golden section search or simple ternary search on t in [0, 1] 
    # minimizing the global covering radius, given that we move vertex `target_v` 
    # along the geodesic to `pole`.
    
    # However, moving just one vertex might not be optimal. 
    # The current program moves all 3. Let's try moving all 3 towards the pole.
    # This preserves the "shape" of the triangle somewhat while shrinking it.
    
    # Let's define the move: new_pts[v] = slerp(pts[v], pole, t) for v in v_indices.
    
    def get_global_radius(pts_mod, t):
        if t == 0.0:
            return current_max_radius
        new_pts = pts_mod.copy()
        for v in v_indices:
            new_pts[v] = _slerp(new_pts[v], pole, t)
            new_pts[v] /= (np.linalg.norm(new_pts[v]) + _EPS)
        
        # Check validity (no coincident points, etc.)
        # Quick check: if any points are too close, return inf
        # We can skip this for speed in the search, and validate at the end.
        
        rad, _, _, _ = _compute_hull_and_radii(new_pts)
        return rad
    
    # Ternary search for t in [0, 1]
    # Note: The function might be non-convex. But usually, moving towards the pole
    # reduces the local facet radius monotonically, but increases neighbor radii.
    # The global max is the envelope.
    
    # Let's try a grid of t values first to find a good region, then refine.
    # Or just a few random samples + the best so far.
    
    # Given the 120s limit, we can afford a small number of hull computations.
    # Let's try t in [0.1, 0.2, ..., 0.9] and pick the best.
    
    best_t = 0.0
    best_r = current_max_radius
    
    # Coarse search
    ts = np.linspace(0.1, 0.9, 9)
    for t in ts:
        r = get_global_radius(pts, t)
        if r < best_r:
            best_r = r
            best_t = t
            
    if best_t > 0:
        # Fine search around best_t
        # Range [best_t - 0.1, best_t + 0.1] clipped to [0, 1]
        t_min = max(0.0, best_t - 0.1)
        t_max = min(1.0, best_t + 0.1)
        ts_fine = np.linspace(t_min, t_max, 9)
        for t in ts_fine:
            r = get_global_radius(pts, t)
            if r < best_r:
                best_r = r
                best_t = t
                
    if best_t > 0:
        new_pts = pts.copy()
        for v in v_indices:
            new_pts[v] = _slerp(new_pts[v], pole, best_t)
            new_pts[v] /= (np.linalg.norm(new_pts[v]) + _EPS)
        return new_pts, best_r
    else:
        return None, current_max_radius

def _random_perturb(pts, magnitude, rng):
    """
    Add random noise to all points to escape local minima.
    """
    noise = rng.standard_normal(pts.shape) * magnitude
    new_pts = pts + noise
    new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
    return new_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci initialization
    best_pts = _initial_fib()
    best_radius, _, _, _ = _compute_hull_and_radii(best_pts)
    
    # Parameters
    perturb_mag = 0.1
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        # Get current worst facet
        curr_radius, worst_idx, v_indices, pole = _compute_hull_and_radii(best_pts)
        
        if v_indices is None:
            break
            
        if curr_radius < best_radius:
            best_radius = curr_radius
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck, try a random kick
        if no_improve_count > max_no_improve:
            kick_pts = _random_perturb(best_pts, 0.5, rng)
            kick_radius, _, _, _ = _compute_hull_and_radii(kick_pts)
            if kick_radius < best_radius:
                best_pts = kick_pts
                best_radius = kick_radius
                no_improve_count = 0
                continue
            else:
                # If kick failed, reduce perturbation magnitude and reset counter
                perturb_mag *= 0.9
                no_improve_count = 0
                if perturb_mag < 0.01:
                    break

        # Try to improve the worst facet
        new_pts, new_radius = _try_improve_worst_facet(best_pts, v_indices, pole, curr_radius, rng)
        
        if new_pts is not None and new_radius < best_radius:
            best_pts = new_pts
            best_radius = new_radius
            no_improve_count = 0
        else:
            # If direct improvement failed, try a small random perturbation
            # focused on the worst facet vertices? Or just global.
            # Let's try a small perturbation of the whole set.
            cand_pts = _random_perturb(best_pts, perturb_mag, rng)
            cand_radius, _, _, _ = _compute_hull_and_radii(cand_pts)
            if cand_radius < best_radius:
                best_pts = cand_pts
                best_radius = cand_radius
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Cool down perturbation
                perturb_mag *= 0.95

    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts /= (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
