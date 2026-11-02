"""
Optimized solver for packing 50 equal spheres in a unit cube.
Strategy:
1. Diverse initializations (FCC, BCC, Random+Relax).
2. Smooth optimization using Log-Sum-Exp (Soft-Min) approximation of the minimum distance.
3. Annealing of the temperature parameter for the Soft-Min.
4. Local search refinement on active constraints.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0  # Safety margin from 150s

def _compute_radius(c):
    """Compute the exact maximum radius for given centers c (N,3)."""
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    # Distance to walls
    # c is in [-0.5, 0.5]
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin_radius(c, t):
    """
    Compute the soft-minimum radius and its gradient.
    Returns: (radius_value, gradient_array)
    """
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    
    # We want to maximize min(d_ij/2, wall_dist_i)
    # Let's collect all constraints:
    # 1. Pair distances / 2
    # 2. Wall distances
    
    # Get upper triangle indices to avoid duplicates
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu] / 2.0
    
    # Wall distances
    wall_dists = (0.5 - np.abs(c)).ravel()
    
    # Combine all constraints
    vals = np.concatenate([pair_dists, wall_dists])
    
    # Softmin calculation: -t * log(sum(exp(-vals/t)))
    # To avoid overflow/underflow, subtract min
    m = np.min(vals)
    if m < 0:
        # Should not happen if initialized well, but just in case
        m = 0
    exp_vals = np.exp(-(vals - m) / t)
    sum_exp = np.sum(exp_vals)
    r_soft = m - t * np.log(sum_exp)
    
    # Gradient calculation
    # d(r_soft)/d(x_k) = - sum_j ( exp(-vals_j/t) / sum_exp ) * d(vals_j)/dx_k
    # weights = exp(-vals/t) / sum_exp
    weights = exp_vals / sum_exp
    
    grad = np.zeros((N, 3))
    
    # Gradient from pairwise distances
    # vals[j] corresponds to pair (i, j) where i < j
    # d(pair_dist_ij) / d(c_k) = (c_i - c_j) / ||c_i - c_j||
    # d(pair_dist_ij / 2) / d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(pair_dist_ij / 2) / d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # We need to accumulate gradients for each point
    # Loop over pairs is O(N^2), which is 2500 for N=50. This is fast enough in Python if vectorized partially.
    # However, doing it in a pure Python loop might be slow. Let's try to vectorize or use a simple loop.
    # For N=50, a simple loop over pairs is ~1250 iterations. Acceptable.
    
    pair_idx = iu
    w_pairs = weights[:len(pair_dists)]
    
    # Vectorized gradient accumulation for pairs is tricky with standard numpy without advanced indexing logic for accumulation.
    # We can use np.add.at
    
    # Compute unit vectors for all pairs
    # diff = c[i] - c[j]
    # dist = ||diff||
    # grad_c_i += 0.5 * w * diff / dist
    # grad_c_j -= 0.5 * w * diff / dist
    
    # To avoid O(N^2) memory for all diffs, we can just loop. 1250 iterations is nothing.
    for k in range(len(pair_dists)):
        i, j = pair_idx[0][k], pair_idx[1][k]
        diff = c[i] - c[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            continue
        grad_vec = 0.5 * w_pairs[k] * diff / dist
        grad[i] += grad_vec
        grad[j] -= grad_vec
        
    # Gradient from wall distances
    # vals[k] for k >= len(pair_dists) corresponds to wall_dist for point p, axis a
    w_walls = weights[len(pair_dists):]
    
    # Wall dist for point p, axis a is 0.5 - |c[p,a]|
    # d(0.5 - |x|)/dx = -sign(x)
    # So d(r)/dx = -w * (-sign(x)) = w * sign(x)
    
    # Map wall indices back to (point, axis)
    # There are N*3 wall constraints.
    # Index k in wall_dists corresponds to point p = k // 3, axis a = k % 3
    # We can vectorize this part.
    
    # Create an array of gradients for walls
    # grad_wall[p, a] = w_walls[k] * sign(c[p, a])
    
    # Reshape weights to (N, 3)
    w_walls_reshaped = w_walls.reshape(N, 3)
    signs = np.sign(c)
    # Handle zero: if c is 0, sign is 0, but derivative of |x| at 0 is undefined. 
    # In practice, points rarely land exactly on 0. If they do, we can pick a random sign or 0.
    # Let's use 0 for now, or 1 if we want to push away.
    # Actually, if c=0, wall dist is 0.5. Derivative is 0 from left, 0 from right? No, |x| derivative is sign(x).
    # If x=0, we can just skip or set to 0.
    grad_wall = w_walls_reshaped * signs
    
    grad += grad_wall
    
    return r_soft, grad

def _optimize_softmin(c_init, deadline, start_t=0.05, end_t=0.001):
    """
    Optimize using soft-min objective with annealing.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    # Log-linear annealing schedule
    # t goes from start_t to end_t
    # Number of steps
    n_steps = 20
    t_schedule = np.logspace(np.log10(start_t), np.log10(end_t), n_steps)
    
    for t in t_schedule:
        if time.time() > deadline:
            break
            
        # Define objective and gradient for this t
        def obj(x):
            r, _ = _softmin_radius(x.reshape(N, 3), t)
            # Add a small penalty if out of bounds to help optimizer stay in range
            # though bounds are handled by L-BFGS
            return -r
        
        def grad(x):
            _, g = _softmin_radius(x.reshape(N, 3), t)
            return -g.flatten()
        
        x0 = c.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj, 
                x0, 
                jac=grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            c = res.x.reshape(N, 3)
            # Clip to ensure validity
            c = np.clip(c, -0.5, 0.5)
            
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
        except Exception as e:
            # If optimizer fails, continue
            pass
            
    return best_c, best_r

def _local_search(c_init, deadline, iterations=5000):
    """
    Local search on active constraints.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    for it in range(iterations):
        if time.time() > deadline:
            break
            
        # Compute distances to find active constraints
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair_d = np.min(d)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_d = np.min(wall_dists)
        
        r = min(min_pair_d / 2.0, min_wall_d)
        
        # Find points involved in the tightest constraints
        active_points = set()
        
        # Tightest pair
        if min_pair_d / 2.0 <= min_wall_d:
            # Find pairs with distance close to min
            threshold = min_pair_d * 1.05
            mask = (d < threshold) & (d > 0)
            pairs = np.argwhere(mask)
            for i, j in pairs:
                if i < j:
                    active_points.add(i)
                    active_points.add(j)
        
        # Tightest walls
        wall_threshold = min_wall_d * 1.05 + 1e-5
        wall_mask = wall_dists < wall_threshold
        # wall_dists is (N, 3). Find indices
        idxs = np.argwhere(wall_mask)
        for i, a in idxs:
            active_points.add(i)
            
        if not active_points:
            active_points = {0} # Fallback
            
        # Pick a random active point to perturb
        active_list = list(active_points)
        idx = active_list[rng.integers(0, len(active_list))]
        
        # Perturb
        # Step size should be small relative to current radius
        step_size = 0.01 * best_r
        c_new = c.copy()
        c_new[idx] += rng.normal(0, step_size, size=3)
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
        else:
            # Sometimes accept worse moves to escape local minima?
            # For now, greedy is safer for the final stage.
            pass
            
    return best_c, best_r

def _init_fcc():
    # Generate FCC lattice points
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx[:N]]
    # Scale to fit
    max_c = np.max(np.abs(pts))
    if max_c > 0:
        pts *= (0.45 / max_c)
    return pts

def _init_bcc():
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx[:N]]
    max_c = np.max(np.abs(pts))
    if max_c > 0:
        pts *= (0.45 / max_c)
    return pts

def _init_random(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.4, 0.4, size=(N, 3))
    # Quick repulsion relaxation to avoid overlaps
    for _ in range(50):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        # Force is inverse square
        inv_d2 = 1.0 / np.maximum(d, 1e-6)**2
        diff = pts[:, None, :] - pts[None, :, :]
        F = np.sum(diff * inv_d2[:, :, None], axis=1)
        pts += 0.01 * F
        pts = np.clip(pts, -0.49, 0.49)
    return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = [
        _init_fcc(),
        _init_bcc(),
        _init_random(1),
        _init_random(2),
        _init_random(3)
    ]
    
    best_pts = None
    best_r = 0.0
    
    # Evaluate initial radii
    for init in inits:
        r = _compute_radius(init)
        if r > best_r:
            best_r = r
            best_pts = init.copy()
            
    # Time allocation:
    # 1. Soft-min optimization on top 3 initializations
    # 2. Local search on the best result
    
    # Sort initializations by radius
    init_r_pairs = []
    for init in inits:
        r = _compute_radius(init)
        init_r_pairs.append((r, init))
    init_r_pairs.sort(reverse=True)
    
    # Optimize the top 3
    for r_init, init in init_r_pairs[:3]:
        if time.time() > start_time + 100: # Leave 45s for local search and safety
            break
        pts, r_opt = _optimize_softmin(init, start_time + 100)
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts
    
    # Local search on the best
    if time.time() < start_time + 140:
        pts, r_ls = _local_search(best_pts, start_time + 140, iterations=10000)
        if r_ls > best_r:
            best_r = r_ls
            best_pts = pts
            
    # Final check
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
        
    return best_pts

# EVOLVE-BLOCK-END
