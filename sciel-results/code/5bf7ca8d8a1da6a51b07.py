# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize, differential_evolution

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
    Smooth surrogate using power means.
    We maximize (M_min / M_max)^2 where M_min approximates the min distance
    and M_max approximates the max distance.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    # Clip to avoid numerical issues with 0 or inf
    d = np.clip(d, 1e-8, 1e8)
    
    # Power mean for min: p -> -inf. Use negative p.
    # M_p = (sum(x^p)/N)^(1/p)
    # For min, we want p to be a large negative number.
    min_mean = (np.sum(d ** (-p_min)) / len(d)) ** (-1.0 / p_min)
    
    # Power mean for max: p -> +inf. Use positive p.
    max_mean = (np.sum(d ** p_max) / len(d)) ** (1.0 / p_max)
    
    if max_mean == 0:
        return 0.0
        
    ratio = min_mean / max_mean
    return ratio ** 2

def _neg_smooth_objective(flat):
    points = flat.reshape(-1, 2)
    return -_smooth_objective(points)

def _neg_true_objective(flat):
    points = flat.reshape(-1, 2)
    return -_compute_score(points)

def _refine(points, time_budget=10.0):
    """
    Refine points using a combination of L-BFGS-B (smooth) and Powell (true).
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Step 1: Quick L-BFGS-B on smooth surrogate
    # This helps navigate the landscape quickly to a better basin.
    if time.time() - start < time_budget * 0.3:
        try:
            res = minimize(_neg_smooth_objective, current_points.ravel(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < -_smooth_objective(current_points):
                current_points = res.x.reshape(-1, 2)
        except Exception:
            pass

    # Step 2: Powell on true objective
    # Powell is derivative-free and robust to the non-smoothness of min/max.
    if time.time() - start < time_budget * 0.8:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < -current_score:
                current_points = res.x.reshape(-1, 2)
                current_score = -res.fun
        except Exception:
            pass

    # Step 3: Final Polish with Nelder-Mead if time permits
    if time.time() - start < time_budget * 0.95:
        try:
            res = minimize(_neg_true_objective, current_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < -current_score:
                current_points = res.x.reshape(-1, 2)
                current_score = -res.fun
        except Exception:
            pass

    return current_points, current_score

def _generate_hexagonal_points(n=16):
    """Generate points in a hexagonal lattice pattern."""
    points = []
    # Hexagonal grid coordinates
    # Using axial coordinates (q, r)
    # Distance in hex grid is related to Euclidean distance
    # We generate a dense hex grid and pick the best n points around the center
    
    # Generate a large hex grid
    q_max = 5
    candidates = []
    for q in range(-q_max, q_max + 1):
        for r in range(-q_max, q_max + 1):
            if abs(q + r) <= q_max:
                # Convert axial to cube
                x = 1.5 * q
                y = np.sqrt(3) * (r + 0.5 * q)
                candidates.append((x, y))
    
    candidates = np.array(candidates)
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Pick 16 points
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
    # Optimal ring radii for 5 and 10 points?
    # 5 points on ring 1 (pentagon), 10 on ring 2 (decagon)
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
    # Use smooth objective for DE
    bounds = [(-3.0, 3.0)] * 32
    de_time_budget = 150
    de_start = time.time()
    
    try:
        # Reduce popsize and maxiter to finish faster, relying on more local refinements
        res = differential_evolution(
            _neg_smooth_objective, 
            bounds, 
            seed=12345,
            maxiter=50,
            tol=1e-8,
            popsize=10,
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
    # We have ~400s left.
    # We will refine the best points and several diverse starts.
    
    # Sort starts by score to prioritize good ones?
    # Or just iterate.
    
    # Ensure best_points is in starts
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
    # Refine the best first with a large budget
    if best_points is not None:
        time_left = deadline - time.time()
        if time_left > 30:
            refined_pts, refined_score = _refine(best_points, time_budget=min(60, time_left - 10))
            _update(refined_pts, refined_score)
            
    # Refine other starts
    for idx, start_pts in enumerate(unique_starts[1:]):
        time_left = deadline - time.time()
        if time_left < 15:
            break
            
        # Allocate remaining time among remaining starts
        num_remaining = len(unique_starts) - 1 - idx
        per_start_time = (time_left - 5) / max(1, num_remaining)
        per_start_time = min(40, per_start_time) # Cap at 40s
        
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
            # Use all remaining time for one final high-precision refinement
            final_pts, final_score = _refine(best_points, time_budget=time_left - 2)
            _update(final_pts, final_score)

    if best_points is None:
        # Fallback
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    return best_points

# EVOLVE-BLOCK-END
