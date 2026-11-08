"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 145.0  # seconds wall-clock limit (5s safety margin from 150s)


def _compute_radii_fast(centers, W, H):
    """Vectorized computation of radii using pairwise distances."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_nn = np.min(dists, axis=1) / 2.0
    cx, cy = centers[:, 0], centers[:, 1]
    boundary_dist = np.minimum.reduce([cx, W - cx, cy, H - cy])
    radii = np.maximum(np.minimum(min_nn, boundary_dist), 1e-12)
    return radii


def _sum_radii(centers, W, H):
    """Objective: sum of radii."""
    return np.sum(_compute_radii_fast(centers, W, H))


def _softmin(values, beta):
    """Stable soft-min via log-sum-exp."""
    min_val = np.min(values)
    exp_terms = np.exp(-beta * (values - min_val))
    sum_exp = np.sum(exp_terms)
    return min_val - np.log(sum_exp) / beta


def _objective_soft(x, W, H, beta):
    """Differentiable soft objective for L-BFGS-B."""
    n = len(x) // 2
    c = x.reshape(n, 2)
    cx, cy = c[:, 0], c[:, 1]
    
    # Distance to boundaries
    d_bound = np.minimum.reduce([cx, W - cx, cy, H - cy])
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    
    total = 0.0
    for i in range(n):
        vals = np.concatenate(([d_bound[i]], half_dists[i, :]))
        total += _softmin(vals, beta)
    
    return -total


def _local_refine(centers, W, H, time_limit=10.0, max_iters=500):
    """Local refinement using L-BFGS-B with soft-min objective, multiple beta levels."""
    n = len(centers)
    if n < 2:
        return centers, _sum_radii(centers, W, H)
    
    best_x = centers.flatten().copy()
    best_val = _sum_radii(centers, W, H)
    
    start_time = time.time()
    betas = [20.0, 50.0, 100.0, 200.0]
    
    bounds = []
    for i in range(n):
        bounds.append((0, W))
        bounds.append((0, H))
    
    for beta in betas:
        if time.time() - start_time > time_limit - 1:
            break
        
        try:
            res = minimize(
                _objective_soft,
                best_x,
                args=(W, H, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < -best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final exact evaluation
    final_centers = best_x.reshape(-1, 2)
    final_val = _sum_radii(final_centers, W, H)
    
    return final_centers, final_val


def _initial_grid(N, W, H):
    """Create an initial grid placement."""
    # For 21 circles, use a 5x5 grid minus 4 corners, or 7x3
    # Try 5x5 grid (25 positions, use 21)
    side = 5
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            centers.append([cx, cy])
    return np.array(centers[:N])


def _initial_grid_7x3(N, W, H):
    """7x3 grid layout."""
    centers = []
    for i in range(3):
        for j in range(7):
            if len(centers) >= N:
                break
            cx = (j + 0.5) * W / 7
            cy = (i + 0.5) * H / 3
            centers.append([cx, cy])
    return np.array(centers[:N])


def _initial_random(N, W, H, seed=None):
    """Create a random initial placement."""
    rng = np.random.default_rng(seed)
    margin = 0.03
    centers = rng.uniform(margin, 1 - margin, size=(N, 2))
    centers[:, 0] *= W
    centers[:, 1] *= H
    return centers


def _optimize_for_aspect(W, H, time_limit=20.0, seed=42):
    """Optimize for a given aspect ratio (W x H)."""
    start_time = time.time()
    best_centers = None
    best_val = -np.inf
    
    def try_optimize(init_centers):
        nonlocal best_centers, best_val
        if time.time() - start_time > time_limit - 1:
            return
        
        val_init = _sum_radii(init_centers, W, H)
        if val_init > best_val:
            best_val = val_init
            best_centers = init_centers.copy()
        
        remaining = time_limit - (time.time() - start_time) - 0.5
        if remaining < 2:
            return
        
        refined, val_ref = _local_refine(init_centers, W, H, 
                                         time_limit=min(remaining, 12),
                                         max_iters=400)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined.copy()
    
    # Try multiple initial configurations
    # 1. 5x5 grid
    try_optimize(_initial_grid(N, W, H))
    
    # 2. 7x3 grid
    try_optimize(_initial_grid_7x3(N, W, H))
    
    # 3. Random starts
    rng = np.random.default_rng(seed)
    n_random = 4
    for i in range(n_random):
        if time.time() - start_time > time_limit - 2:
            break
        centers_rand = _initial_random(N, W, H, seed=seed + i * 1000)
        try_optimize(centers_rand)
    
    return best_centers, best_val


def construct():
    """Main entry point: pack 21 circles in a perimeter-4 rectangle, maximize sum of radii."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_centers = None
    best_val = 0.0
    best_W = None
    best_H = None
    
    # Perimeter constraint: 2*(W+H) = 4 => W+H = 2
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.18, 0.8, 1.25]
    
    # Allocate time: more time for square, less for others
    total_time = 130.0
    n_ar = len(aspect_ratios)
    # Weight: square gets more time
    weights = [0.25, 0.15, 0.15, 0.12, 0.12, 0.10, 0.11]
    
    for idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline - 5.0:
            break
        
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        remaining = deadline - time.time() - 3.0
        if remaining < 4:
            break
        
        time_alloc = min(remaining * weights[idx] * 2.5, 25.0)
        time_alloc = max(time_alloc, 4.0)
        
        centers, val = _optimize_for_aspect(W, H, time_limit=time_alloc, seed=42 + idx * 7)
        
        if centers is not None and val > best_val:
            best_val = val
            best_centers = centers.copy()
            best_W = W
            best_H = H
    
    if best_centers is None:
        W, H = 1.0, 1.0
        best_centers = _initial_grid(N, W, H)
        best_W, best_H = W, H
        best_val = _sum_radii(best_centers, best_W, best_H)
    
    # Final polish with high precision
    remaining = deadline - time.time()
    if remaining > 3:
        refined, val_ref = _local_refine(best_centers, best_W, best_H,
                                         time_limit=min(remaining - 1, 15),
                                         max_iters=1000)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined
    
    # One more polish pass
    remaining = deadline - time.time()
    if remaining > 2:
        refined, val_ref = _local_refine(best_centers, best_W, best_H,
                                         time_limit=min(remaining - 0.5, 8),
                                         max_iters=500)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined
    
    radii = _compute_radii_fast(best_centers, best_W, best_H)
    radii = radii * (1 - 1e-7)
    
    result = np.column_stack([best_centers[:, 0], best_centers[:, 1], radii])
    
    return result


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Number of circles: {len(result)}")
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"First few circles: {result[:3]}")
