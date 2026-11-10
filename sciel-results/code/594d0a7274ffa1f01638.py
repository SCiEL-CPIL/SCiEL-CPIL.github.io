# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize, differential_evolution, LinearConstraint

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if points.shape[0] < 2:
        return 0.0
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(points, p_min=10.0, p_max=10.0):
    """
    Smooth surrogate using log-sum-exp based soft min/max.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    # Clip to avoid numerical issues
    d = np.clip(d, 1e-10, 1e10)
    
    # Soft Min
    d_neg = -d
    max_d_neg = np.max(d_neg)
    log_sum_exp_neg = max_d_neg + np.log(np.sum(np.exp(d_neg - max_d_neg)))
    soft_min = -log_sum_exp_neg / p_min
    
    # Soft Max
    d_pos = d
    max_d_pos = np.max(d_pos)
    log_sum_exp_pos = max_d_pos + np.log(np.sum(np.exp(d_pos - max_d_pos)))
    soft_max = log_sum_exp_pos / p_max
    
    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio ** 2

def _neg_smooth_objective(flat):
    points = flat.reshape(-1, 2)
    return -_smooth_objective(points)

def _neg_true_objective(flat):
    points = flat.reshape(-1, 2)
    return -_compute_score(points)

def _sqp_refine(points, time_budget=10.0):
    """
    Refine using SQP-style constrained optimization.
    We fix the scale such that d_max = 1.0 (by normalizing the best point set).
    Then we maximize d_min subject to d_ij <= 1.0 for all pairs.
    This is equivalent to maximizing the ratio directly in the normalized space.
    
    Actually, simpler: 
    1. Normalize points so that d_max = 1.
    2. Maximize min(d_ij) subject to d_ij <= 1.
    3. The resulting min(d_ij) is the optimal ratio.
    4. Scale back? No, the ratio is invariant to scaling.
    So we just return the points in the normalized space (or any scaling).
    The score is (d_min/d_max)^2. If we normalize d_max=1, score is d_min^2.
    """
    start = time.time()
    current_points = points.copy()
    
    # Normalize to unit max distance
    d = pdist(current_points)
    d_max = np.max(d)
    if d_max > 0:
        current_points = current_points / d_max
    
    # Now d_max is approx 1. We want to maximize d_min subject to d_ij <= 1.
    # Objective: maximize min(d_ij)
    # Constraints: d_ij^2 <= 1 for all i<j
    
    n = current_points.shape[0]
    # Indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def obj(x):
        pts = x.reshape(n, 2)
        # We want to maximize min distance, so minimize -min distance
        # But min is non-smooth. Use a smooth approximation or just use the true min
        # For SQP, we need a smooth objective. 
        # Let's try maximizing the product of distances? No, that's different.
        # Let's use a penalty method or just trust SLSQP with a smooth surrogate for min.
        # Actually, SLSQP can handle non-smooth objectives if we are careful, but it's not recommended.
        # Better: Minimize -soft_min with constraints d_ij^2 <= 1.
        # Or simply: Maximize min distance.
        # Let's stick to the smooth surrogate for the objective in SLSQP, 
        # but enforce the hard constraint d_ij^2 <= 1.
        pts = x.reshape(n, 2)
        d = pdist(pts)
        # Soft min approximation for smoothness
        p = 20.0
        d_clipped = np.clip(d, 1e-8, 1e8)
        soft_min = (np.sum(d_clipped ** (-p)) / len(d_clipped)) ** (-1.0/p)
        return -soft_min

    # Constraints: d_ij^2 <= 1
    # SLSQP constraints are of the form g(x) >= 0
    # So we need 1 - d_ij^2 >= 0
    
    def make_constraint(i, j):
        def con(x):
            pts = x.reshape(n, 2)
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            return 1.0 - (dx*dx + dy*dy)
        return con

    # This is too many constraints for SLSQP (120 constraints).
    # SLSQP might struggle.
    # Alternative: Use a penalty method with L-BFGS-B.
    # Minimize -soft_min + lambda * sum(max(0, d_ij^2 - 1)^2)
    
    def penalized_obj(x, lam=10.0):
        pts = x.reshape(n, 2)
        d = pdist(pts)
        d_clipped = np.clip(d, 1e-8, 1e8)
        p = 20.0
        soft_min = (np.sum(d_clipped ** (-p)) / len(d_clipped)) ** (-1.0/p)
        
        # Penalty for violating d_ij <= 1
        viol = np.maximum(0, d - 1.0)
        penalty = lam * np.sum(viol ** 2)
        
        return -soft_min + penalty

    # Start with current points (already normalized)
    x0 = current_points.ravel()
    
    best_x = x0.copy()
    best_val = penalized_obj(x0)
    
    # Iteratively increase penalty
    lam = 10.0
    for _ in range(5):
        if time.time() - start > time_budget:
            break
        
        try:
            res = minimize(penalized_obj, x0, method='L-BFGS-B', args=(lam,),
                           options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
            x0 = res.x
        except Exception:
            break
        
        lam *= 5.0

    refined_points = best_x.reshape(n, 2)
    
    # Ensure we didn't drift too far or become invalid
    d_ref = pdist(refined_points)
    if np.any(d_ref < 1e-6) or np.max(d_ref) > 1.1: # Tolerance for scaling
        return current_points, _compute_score(current_points)
        
    score = _compute_score(refined_points)
    return refined_points, score

def _refine(points, time_budget=10.0):
    """
    Refine points using a combination of L-BFGS-B (smooth), SQP-like penalty, and Powell (true).
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Step 1: Quick L-BFGS-B on smooth surrogate
    if time.time() - start < time_budget * 0.2:
        try:
            res = minimize(_neg_smooth_objective, current_points.ravel(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < -_smooth_objective(current_points):
                current_points = res.x.reshape(-1, 2)
                current_score = _compute_score(current_points)
        except Exception:
            pass

    # Step 2: SQP-like refinement (Penalty method for max-min with bounds)
    if time.time() - start < time_budget * 0.5:
        try:
            sqp_pts, sqp_score = _sqp_refine(current_points, time_budget=min(20, time_budget * 0.3))
            if sqp_score > current_score:
                current_points = sqp_pts
                current_score = sqp_score
        except Exception:
            pass

    # Step 3: Powell on true objective
    if time.time() - start < time_budget * 0.8:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < -current_score:
                current_points = res.x.reshape(-1, 2)
                current_score = -res.fun
        except Exception:
            pass

    # Step 4: Final Polish with Nelder-Mead if time permits
    if time.time() - start < time_budget * 0.95:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < -current_score:
                current_points = res.x.reshape(-1, 2)
                current_score = -res.fun
        except Exception:
            pass

    return current_points, current_score

def _generate_hexagonal_points(n=16):
    """Generate points in a hexagonal lattice pattern."""
    candidates = []
    q_max = 5
    for q in range(-q_max, q_max + 1):
        for r in range(-q_max, q_max + 1):
            if abs(q + r) <= q_max:
                x = 1.5 * q
                y = np.sqrt(3) * (r + 0.5 * q)
                candidates.append((x, y))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists)
    selected = candidates[sorted_indices[:n]]
    return selected

def _generate_ring_points(radii, counts, angles_offsets=None):
    """Generate points on concentric rings."""
    points = []
    total = sum(counts)
    if total != 16:
        return None
    
    for i, (r, c) in enumerate(zip(radii, counts)):
        if c == 0:
            continue
        if c == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, c, endpoint=False)
            if angles_offsets and i < len(angles_offsets):
                angles += angles_offsets[i]
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points.extend(zip(x, y))
            
    return np.array(points)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()

    # 1. Initializations
    starts = []
    
    # 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    starts.append(grid_points)
    
    # Circle
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(circle_points)
    
    # Hexagonal
    hex_points = _generate_hexagonal_points(16)
    starts.append(hex_points)
    
    # Rings: 1 center, 5 inner, 10 outer
    p_rings1 = _generate_ring_points([0.8, 1.8], [1, 5, 10], [0, 0, np.pi/10])
    if p_rings1 is not None:
        starts.append(p_rings1)
        
    # Rings: 1 center, 6 inner, 9 outer
    p_rings2 = _generate_ring_points([0.7, 1.6], [1, 6, 9], [0, 0, np.pi/9])
    if p_rings2 is not None:
        starts.append(p_rings2)
        
    # Random
    starts.append(rng.uniform(-2, 2, size=(16, 2)))
    starts.append(rng.uniform(-2, 2, size=(16, 2)))
    
    # Update best with initializations
    for s in starts:
        if s is not None and s.shape == (16, 2):
            sc = _compute_score(s)
            _update(s, sc)
    
    # 2. Differential Evolution (Global Search)
    bounds = [(-3.0, 3.0)] * 32
    
    try:
        res = differential_evolution(
            _neg_smooth_objective, 
            bounds, 
            seed=12345,
            maxiter=80,
            tol=1e-10,
            popsize=15,
            polish=False,
            init='sobol'
        )
        
        candidate_points = res.x.reshape(-1, 2)
        candidate_score = _compute_score(candidate_points)
        _update(candidate_points, candidate_score)
        
        # Also try the smooth optimum as a start for refinement
        starts.append(candidate_points)
        
    except Exception:
        pass
        
    # 3. Local Refinement Loop
    
    if best_points is not None:
        starts.insert(0, best_points)
        
    # Remove duplicates
    unique_starts = []
    for s in starts:
        if s is None or s.shape != (16, 2):
            continue
        is_dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
            
    # Prioritize the current best
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 30:
            refined_pts, refined_score = _refine(best_points, time_budget=min(80, time_left - 10))
            _update(refined_pts, refined_score)
            
    # Refine other starts
    for idx, start_pts in enumerate(unique_starts[1:]):
        time_left = deadline - time.time()
        if time_left < 15:
            break
            
        num_remaining = len(unique_starts) - 1 - idx
        per_start_time = (time_left - 5) / max(1, num_remaining)
        per_start_time = min(50, per_start_time)
        
        if per_start_time < 5:
            continue
            
        try:
            refined_pts, refined_score = _refine(start_pts, time_budget=per_start_time)
            _update(refined_pts, refined_score)
        except Exception:
            pass

    # 4. Final Polish
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 10:
            final_pts, final_score = _refine(best_points, time_budget=time_left - 2)
            _update(final_pts, final_score)

    if best_points is None:
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    return best_points

# EVOLVE-BLOCK-END
