"""Combined program: grid initialization + L-BFGS-B soft-min refinement + local perturbation polish."""
import numpy as np
from scipy.optimize import minimize
import time as _time

N = 100

# EVOLVE-BLOCK-START
_DEADLINE = _time.time() + 140.0  # 150s wall-clock limit, keep 10s safety margin
_N = 100

def _compute_radius(centers):
    """Compute the largest common radius for given centers inside [-0.5, 0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (_N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_objective(x, beta):
    """
    Objective for L-BFGS-B: minimize -soft_min of all radius-limiting distances.
    The soft-min smoothly approximates the true min, making the objective differentiable.
    """
    centers = x.reshape(_N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    
    # Collect all radius-limiting values: d_ij/2 for i<j, and wall distances
    iu = np.triu_indices(_N, k=1)
    pair_dists = dists[iu] / 2.0
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Stable log-sum-exp based soft-min
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    # For numerical stability, cap the exponent argument
    exp_args = -beta * shifted
    exp_args = np.maximum(exp_args, -700.0)  # avoid underflow
    log_sum = np.log(np.sum(np.exp(exp_args)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    
    return -soft_min_val

def _build_grid(m1, m2, m3, jitter=0.0):
    """Build a grid of size m1 x m2 x m3 and return first N points."""
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        noise = np.random.randn(*grid.shape) * jitter
        grid = grid + noise
        np.clip(grid, -0.499, 0.499, out=grid)
    return grid[:_N]

def _lbfgsb_refine(centers, max_time=60.0):
    """
    Refine centers using L-BFGS-B with an adaptive beta (soft-min sharpness) schedule.
    Starts with a smooth approximation and gradually sharpens to approach the true min.
    """
    start = _time.time()
    x = centers.flatten().copy()
    bounds = [(-0.4999, 0.4999)] * (_N * 3)
    
    # Adaptive beta schedule: start smooth, end sharp
    betas = [10, 30, 100, 300, 1000, 3000]
    
    for beta in betas:
        if _time.time() > _DEADLINE - 15 or (_time.time() - start) > max_time:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
            )
            
            new_centers = res.x.reshape(_N, 3)
            # Validate: ensure all points are inside the cube
            if np.all(np.abs(new_centers) <= 0.4999) and np.all(np.isfinite(new_centers)):
                x = new_centers.flatten()
            else:
                # Clip back if optimizer pushed outside
                new_centers = np.clip(res.x.reshape(_N, 3), -0.4999, 0.4999)
                x = new_centers.flatten()
                
        except Exception:
            pass
    
    return x.reshape(_N, 3)

def _local_perturbation_refine(centers, max_time=30.0):
    """
    Per-point local refinement: try small random perturbations on each point,
    keep if it improves the global minimum radius.
    """
    start = _time.time()
    centers = centers.copy()
    n = centers.shape[0]
    r = _compute_radius(centers)
    
    # Multiple passes over all points
    for _pass in range(10):
        if _time.time() > _DEADLINE - 10 or (_time.time() - start) > max_time:
            break
        
        for i in range(n):
            if _time.time() > _DEADLINE - 10 or (_time.time() - start) > max_time:
                break
            
            orig = centers[i].copy()
            best_local = orig.copy()
            best_r = r
            
            # Try several random perturbations of decreasing magnitude
            for scale in [0.01, 0.005, 0.002, 0.001]:
                for _ in range(3):
                    perturb = np.random.randn(3) * scale
                    new_pt = centers[i] + perturb
                    if np.any(np.abs(new_pt) > 0.4999):
                        continue
                    centers[i] = new_pt
                    new_r = _compute_radius(centers)
                    if new_r > best_r + 1e-15:
                        best_r = new_r
                        best_local = new_pt.copy()
                    else:
                        centers[i] = best_local
            
            centers[i] = best_local
            r = best_r
    
    return centers

def construct():
    start_time = _time.time()
    
    # Try multiple grid configurations and pick the best initial
    grid_configs = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (4, 4, 6),
        (6, 4, 4),
        (4, 6, 4),
        (5, 5, 4),  # repeat with jitter below
    ]
    
    best_centers = None
    best_r = -np.inf
    
    for m1, m2, m3 in grid_configs:
        if _time.time() > _DEADLINE - 20:
            break
        if m1 * m2 * m3 < _N:
            continue
        c = _build_grid(m1, m2, m3, jitter=0.0)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_centers = c.copy()
    
    # Also try a jittered version of the best grid
    if best_centers is not None and _time.time() < _DEADLINE - 20:
        np.random.seed(42)
        # Rebuild the grid that gave best_r with small jitter
        # Just jitter the current best
        c_jit = best_centers + np.random.randn(*best_centers.shape) * 0.005
        np.clip(c_jit, -0.499, 0.499, out=c_jit)
        r_jit = _compute_radius(c_jit)
        if r_jit > best_r:
            best_r = r_jit
            best_centers = c_jit.copy()
    
    # Phase 1: L-BFGS-B soft-min refinement (main optimization)
    if best_centers is not None and _time.time() < _DEADLINE - 20:
        try:
            refined = _lbfgsb_refine(best_centers, max_time=80.0)
            r_ref = _compute_radius(refined)
            if r_ref > best_r:
                best_r = r_ref
                best_centers = refined
        except Exception:
            pass
    
    # Phase 2: Local perturbation refinement (polish)
    if best_centers is not None and _time.time() < _DEADLINE - 10:
        try:
            polished = _local_perturbation_refine(best_centers, max_time=25.0)
            r_pol = _compute_radius(polished)
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished
        except Exception:
            pass
    
    # Final validation
    if best_centers is None:
        # Fallback: simple grid
        best_centers = _build_grid(5, 5, 4, jitter=0.0)
    
    final_r = _compute_radius(best_centers)
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        best_centers = _build_grid(5, 5, 4, jitter=0.0)
    
    return best_centers
# EVOLVE-BLOCK-END
