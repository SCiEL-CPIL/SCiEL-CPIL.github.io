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

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_data(pts):
    """
    Compute convex hull facets, their normals (poles), and angular circumradii.
    Returns:
        radii: array of angular circumradii for each valid facet
        normals: array of unit normals (poles) for each valid facet
        faces: array of vertex indices for each valid facet
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Vectors for the triangle edges
        AB = B - A
        AC = C - A
        
        # Normal vector for each facet (unnormalized)
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Handle degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([])
            
        faces_valid = faces[valid]
        A_v = A[valid]
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward (away from origin)
        # We can check by dotting with a vector from origin to centroid
        centroids = (A_v + pts[faces_valid[:, 1]] + pts[faces_valid[:, 2]]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: distance from pole to any vertex (e.g., A)
        # cos(theta) = P @ A
        cos_theta = np.sum(normals * A_v, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return ang_radii, normals, faces_valid
    except Exception:
        return np.array([]), np.array([]), np.array([])

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    radii, _, _ = _compute_hull_data(pts)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def _move_point_on_sphere(p, target, step):
    """
    Move point p towards target on the unit sphere by a geodesic step size.
    p, target: unit vectors.
    step: angular step size in radians.
    """
    # Normalize target just in case
    t_norm = np.linalg.norm(target)
    if t_norm < _EPS:
        return p
    target = target / t_norm
    
    cos_theta = np.clip(np.dot(p, target), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if theta < _EPS:
        return p
    
    # Slerp
    sin_theta = np.sin(theta)
    if sin_theta < _EPS:
        # If very close, just move linearly and normalize
        new_p = p + (target - p) * (step / theta)
        return new_p / (np.linalg.norm(new_p) + _EPS)
    
    # We want to move by 'step' radians from p towards target
    # If step > theta, we overshoot, but usually step is small
    s = step / theta
    if s > 1.0:
        s = 1.0 # Don't go past target
        
    # slerp formula
    # p_new = (sin((1-s)*theta)/sin(theta)) * p + (sin(s*theta)/sin(theta)) * target
    w1 = np.sin((1.0 - s) * theta) / sin_theta
    w2 = np.sin(s * theta) / sin_theta
    new_p = w1 * p + w2 * target
    new_p /= (np.linalg.norm(new_p) + _EPS)
    return new_p

def _worst_facet_refinement(pts, n_iters, deadline):
    """
    Iteratively find the worst facet and move its vertices towards its pole.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    step_size = 0.1
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
            
        radii, normals, faces = _compute_hull_data(pts)
        if len(radii) == 0:
            break
            
        curr_r = np.max(radii)
        
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
        # Find the worst facet
        worst_idx = np.argmax(radii)
        worst_face = faces[worst_idx]
        pole = normals[worst_idx]
        
        # Try to improve by moving the 3 vertices towards the pole
        # We do a local line search for the optimal step size for this specific move
        
        # Define a direction for each vertex: towards the pole
        # But moving towards the pole might increase other radii.
        # We simply try a few step sizes and keep the best.
        
        original_pts = pts.copy()
        found_improvement = False
        best_step = 0.0
        best_local_r = curr_r
        
        # Try a range of step sizes
        # Adaptive step: start small, increase if improving
        steps_to_try = [step_size * 0.1, step_size * 0.3, step_size * 0.6, step_size, step_size * 1.5, step_size * 2.0]
        
        for s in steps_to_try:
            cand = original_pts.copy()
            for v in worst_face:
                cand[v] = _move_point_on_sphere(cand[v], pole, s)
            
            cand_r = _compute_hull_radius(cand)
            if cand_r < best_local_r:
                best_local_r = cand_r
                best_step = s
                found_improvement = True
                
        if found_improvement:
            # Apply the best step
            for v in worst_face:
                pts[v] = _move_point_on_sphere(pts[v], pole, best_step)
            # Keep step size if it worked well, maybe increase slightly?
            # For now, keep it stable or slightly increase
            step_size = max(0.01, min(0.5, step_size * 1.1))
        else:
            # If no improvement, reduce step size
            step_size *= 0.7
            if step_size < 1e-5:
                # Stuck on this facet? Try a random perturbation of the whole config?
                # Or just break and let the outer loop handle it.
                # Let's try a small random kick to the worst vertices to escape local min
                rng = np.random.default_rng(hash(tuple(pts.flatten().round(8))))
                for v in worst_face:
                    noise = rng.standard_normal(3) * 0.01
                    cand_v = pts[v] + noise
                    cand_v /= (np.linalg.norm(cand_v) + _EPS)
                    pts[v] = cand_v
                step_size = 0.1 # Reset step after kick
                
    return best_pts, best_r

def _random_restart_polish(pts, n_iters, deadline):
    """
    Perform a few random restarts of the refinement process to escape local minima.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    time_per_restart = (deadline - time.time()) / (n_restarts + 1)
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb current best
        cand = best_pts.copy()
        # Small random perturbation
        noise = rng.standard_normal((N, 3)) * 0.1
        cand += noise
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        
        # Run refinement on the perturbed version
        refined, r = _worst_facet_refinement(cand, n_iters=50, deadline=deadline - 5.0)
        
        if r < best_r:
            best_r = r
            best_pts = refined
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Fibonacci spiral
    pts = _initial_fib()
    
    # Initial refinement
    pts, best_r = _worst_facet_refinement(pts, n_iters=200, deadline=deadline - 60.0)
    
    # Random restarts to escape local minima
    pts, best_r = _random_restart_polish(pts, n_iters=100, deadline=deadline - 10.0)
    
    # Final polish on the best found
    if time.time() < deadline - 2.0:
        pts, best_r = _worst_facet_refinement(pts, n_iters=200, deadline=deadline - 1.0)
        
    # Final normalization and validity check
    pts = np.array(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < _EPS] = 1.0
    pts /= norms
    
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return _initial_fib()
        
    return pts
