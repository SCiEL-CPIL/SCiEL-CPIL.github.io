# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _compute_areas(points):
    """Compute all 286 triangle areas for 13 points."""
    n = len(points)
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return areas, (i_flat, j_flat, k_flat)

def _score_fast(points):
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    areas, _ = _compute_areas(points)
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _soft_min_score(points, temperature=0.01):
    """
    Compute a smooth surrogate for A_min / A_hull using soft-min via log-sum-exp.
    This provides well-defined gradients for optimization.
    """
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    areas, _ = _compute_areas(points)
    
    # Soft-min: -temperature * log(sum(exp(-areas/temperature)))
    # As temperature -> 0, this approaches the true min
    shifted = -areas / temperature
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_area = -temperature * log_sum_exp
    
    if soft_min_area <= 1e-12:
        return 0.0
    return soft_min_area / a_hull

def _negative_score_smooth(points, temperature=0.01):
    """Negative smooth score for minimization."""
    s = _soft_min_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    
    Uses smooth soft-min surrogate for gradient-based optimization, then
    refines with high-precision polish passes.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    n = 13
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(seed=42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular 13-gon (circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)
    
    # 2. Hexagonal lattice pattern
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 3. Grid-based (4x4 minus one corner)
    grid_pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.5, 0.5], [1.5, 0.5],
        [0.5, 1.5], [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Two concentric circles
    inner_r = 0.4
    outer_r = 1.0
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_angles = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 6, endpoint=False)
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
    outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
    concentric = np.vstack([inner_pts, outer_pts])
    initializations.append(concentric)
    
    # 5. Random points in unit square (multiple seeds)
    for seed in [1, 2, 3, 4, 5]:
        rng_temp = np.random.default_rng(seed=seed)
        initializations.append(rng_temp.random((n, 2)))
    
    # 6. Perturbed regular polygon
    for perturb_scale in [0.1, 0.2, 0.3]:
        perturbed = reg_pts + rng.normal(size=(n, 2)) * perturb_scale
        initializations.append(perturbed)
    
    # Evaluate initializations and pick top ones
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 60:
            break
        s = _score_fast(init_pts)
        scored_inits.append((s, init_pts))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Take top 6 initializations for optimization (one more than before)
    top_inits = [pts for _, pts in scored_inits[:6]]
    
    # Gradient-based optimization using smooth soft-min surrogate
    for init_idx, init_pts in enumerate(top_inits):
        if time.time() > deadline - 40:
            break
        
        current = init_pts.copy()
        cur_score = _score_fast(current)
        update_best(current, cur_score)
        
        # Use smooth objective with annealed temperature
        for temp in [0.05, 0.02, 0.005, 0.001]:
            if time.time() > deadline - 30:
                break
            
            def objective_smooth(flat_pts, t=temp):
                pts = flat_pts.reshape(n, 2)
                return _negative_score_smooth(pts, t)
            
            x0 = current.flatten()
            
            try:
                result = minimize(
                    objective_smooth,
                    x0,
                    method='BFGS',
                    options={'maxiter': 400, 'gtol': 1e-7}
                )
                
                new_pts = result.x.reshape(n, 2)
                new_score = _score_fast(new_pts)
                if new_score > cur_score:
                    current = new_pts
                    cur_score = new_score
                    update_best(current, cur_score)
            except Exception:
                pass
        
        # L-BFGS-B refinement with very small temperature
        if time.time() > deadline - 30:
            break
        
        try:
            def objective_tight(flat_pts):
                pts = flat_pts.reshape(n, 2)
                return _negative_score_smooth(pts, 0.0005)
            
            result2 = minimize(
                objective_tight,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-9}
            )
            
            new_pts = result2.x.reshape(n, 2)
            new_score = _score_fast(new_pts)
            if new_score > cur_score:
                current = new_pts
                cur_score = new_score
                update_best(current, cur_score)
        except Exception:
            pass
    
    # Additional random restarts from best point with small perturbations
    for restart in range(12):
        if time.time() > deadline - 30:
            break
        
        perturbed = best_points + rng.normal(size=(n, 2)) * 0.03
        s = _score_fast(perturbed)
        if s > best_score:
            best_points = perturbed.copy()
            best_score = s
        
        # Quick optimization from here with smooth objective
        try:
            def objective_restart(flat_pts):
                pts = flat_pts.reshape(n, 2)
                return _negative_score_smooth(pts, 0.01)
            
            result = minimize(
                objective_restart,
                best_points.flatten(),
                method='BFGS',
                options={'maxiter': 250, 'gtol': 1e-8}
            )
            
            new_pts = result.x.reshape(n, 2)
            new_score = _score_fast(new_pts)
            if new_score > best_score:
                best_points = new_pts
                best_score = new_score
        except Exception:
            pass
    
    # Final high-precision polish pass
    if best_points is not None and time.time() < deadline - 10:
        try:
            def objective_polish(flat_pts):
                pts = flat_pts.reshape(n, 2)
                return _negative_score_smooth(pts, 0.0001)
            
            result_polish = minimize(
                objective_polish,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            
            new_pts = result_polish.x.reshape(n, 2)
            new_score = _score_fast(new_pts)
            if new_score > best_score:
                best_points = new_pts
                best_score = new_score
        except Exception:
            pass
    
    # Final validation
    final_score = _score_fast(best_points)
    if final_score <= 0 or not np.all(np.isfinite(best_points)):
        # Fallback to a known valid configuration
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
