import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _compute_radius(points):
    """Compute the exact minimum radius for a set of points."""
    # Pairwise distances
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min() / 2.0
    
    # Wall clearances
    clearance = (0.5 - np.abs(points)).min()
    
    return min(min_pair, clearance)

def _softmin_obj_and_grad(points_flat, alpha=50.0):
    """
    Compute the smooth objective function and its gradient.
    Objective: -softmin(clearances)
    Clearances: half pairwise distances and wall clearances.
    """
    points = points_flat.reshape(N, 3)
    
    # Pairwise distances
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-8, dists, 1e-8)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    pair_vecs = d[iu] / (2.0 * dists_safe[iu, None]) # Gradient of dist/2 wrt p_i is (p_i-p_j)/(2*dist)
    
    # Wall clearances
    wall_clearances = 0.5 - np.abs(points)
    wall_grads = -np.sign(points) # Gradient of (0.5 - |x|) is -sign(x)
    
    # Combine clearances
    all_clearances = np.concatenate([pair_dists, wall_clearances.flatten()])
    
    # Softmin calculation
    # S = sum(exp(-alpha * c))
    exp_terms = np.exp(-alpha * all_clearances)
    S = np.sum(exp_terms)
    
    # Objective: -1/alpha * log(S)
    obj = -1.0 / alpha * np.log(S)
    
    # Gradient: sum( w_k * grad_c_k ) where w_k = exp(-alpha*c_k)/S
    weights = exp_terms / S
    
    grad = np.zeros_like(points)
    
    # Pairwise contributions
    # For each pair (i, j), weight w
    # grad_p_i += w * (p_i - p_j) / (2 * dist)
    # grad_p_j += w * (p_j - p_i) / (2 * dist)
    
    # We have weights for pairs in weights[:len(pair_dists)]
    pair_weights = weights[:len(pair_dists)]
    
    # Accumulate gradients for pairs
    # We can use a loop or vectorized accumulation. Loop is fine for 4950 pairs.
    # But vectorized is faster.
    # Let's use a helper to scatter add.
    # Actually, for N=100, a simple loop over pairs might be acceptable if not too slow, 
    # but vectorized is better.
    
    # Create arrays of indices
    i_idx, j_idx = iu
    
    # For each pair, we add to grad[i] and grad[j]
    # grad[i] += pair_weights[:, None] * pair_vecs
    # grad[j] -= pair_weights[:, None] * pair_vecs
    
    # Using np.add.at for efficiency
    np.add.at(grad, i_idx, pair_weights[:, None] * pair_vecs)
    np.add.at(grad, j_idx, -pair_weights[:, None] * pair_vecs)
    
    # Wall contributions
    # For each point i and coord k, weight w
    # grad[i, k] += w * (-sign(p[i, k]))
    wall_weights = weights[len(pair_dists):].reshape(N, 3)
    grad += wall_weights * wall_grads
    
    return obj, grad.flatten()

def _optimize_from_init(init_points, alpha=50.0, max_iter=500):
    """Run L-BFGS-B optimization from a given initial point."""
    x0 = init_points.flatten()
    
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    res = minimize(
        _softmin_obj_and_grad,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    best_pts = res.x.reshape(N, 3)
    # Ensure strictly inside
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pts

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin
    
    best_pts = None
    best_r = -1.0
    
    # Strategy: Multiple restarts with different seeds
    # 1. Simple cubic grid (5x5x4)
    # 2. Jittered grid
    # 3. Random
    
    rng = np.random.default_rng(42)
    
    # Seed 1: 5x5x4 Grid
    # 5*5*4 = 100 points
    # x, y in [-0.4, 0.4] with step 0.2 -> 5 points
    # z in [-0.375, 0.375] with step 0.25 -> 4 points? 
    # Let's just use a 5x5x4 grid centered.
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    grid_pts = np.array([(x, y, z) for z in z for y in y for x in x])
    
    # Seed 2: Jittered Grid
    jitter_pts = grid_pts + rng.uniform(-0.05, 0.05, size=grid_pts.shape)
    jitter_pts = np.clip(jitter_pts, -0.45, 0.45)
    
    # Seed 3: Random
    random_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
    
    seeds = [grid_pts, jitter_pts, random_pts]
    
    # We can add more random seeds if time permits
    for i in range(5):
        if time.time() - t0 > DEADLINE - 20:
            break
        seeds.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
        
    for seed_pts in seeds:
        if time.time() - t0 > DEADLINE:
            break
            
        # Run optimization
        # Use a moderate alpha. If alpha is too high, it's like hard min (non-smooth).
        # If too low, it's too smooth (averages out). 50-100 is good.
        try:
            opt_pts = _optimize_from_init(seed_pts, alpha=80.0, max_iter=200)
            r = _compute_radius(opt_pts)
            if r > best_r:
                best_r = r
                best_pts = opt_pts
        except Exception:
            pass
            
    # If no valid points found (shouldn't happen), fallback to grid
    if best_pts is None:
        best_pts = grid_pts
    
    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid_pts
        
    return best_pts
