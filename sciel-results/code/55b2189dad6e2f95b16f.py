"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _sa_search(centers, rng, budget_time):
    """
    Simulated annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.05
    T_end = 1e-6
    max_steps = 500000
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Random perturbation
        idx = rng.integers(0, N)
        # Step size decays over time
        step_size = 0.02 * (1.0 + 0.5 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            # Accept bad moves with probability exp(delta_r / T)
            if delta_r > -10 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def _soft_min_obj(x, tau):
    """
    Soft-min objective: minimize -soft_min(all distances).
    x: flattened array of 2*N coordinates
    tau: temperature parameter for soft-min
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Collect all relevant distances (pairwise and wall)
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([walls, pair_dists[np.triu_indices(n, k=1)]])
    
    # Soft-min: -log(sum(exp(-d/tau))) * tau
    # This is a smooth approximation to min(all_dists)
    min_d = np.min(all_dists)
    # Use stable log-sum-exp
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _soft_min_grad(x, tau):
    """
    Gradient of the soft-min objective.
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([walls, pair_dists[np.triu_indices(n, k=1)]])
    
    # Soft-min gradient
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # d/dx_i wall_dist = -sign(x_i) if wall_dist_x < wall_dist_y else 0 (for x)
    # Actually, wall_dist = min(0.5 - |x|, 0.5 - |y|)
    # Let's just use finite differences or careful derivation
    # For simplicity and robustness, let's use a simple numerical gradient if needed, 
    # but L-BFGS-B can handle it. However, providing an analytical gradient is faster.
    
    # Let's stick to the objective function for L-BFGS-B which does numerical differentiation,
    # or implement the gradient carefully.
    # Given time constraints, let's rely on L-BFGS-B's numerical gradient first, 
    # or implement a simple one.
    
    # Actually, let's just return the objective and let scipy handle it, 
    # or implement a basic gradient.
    
    # Gradient of soft_min = min_d - tau * log(sum(exp(-(d_i - min_d)/tau)))
    # d(soft_min)/d(x) = sum_j w_j * d(d_j)/d(x)
    # where w_j = exp(-(d_j - min_d)/tau) / sum(exp(-(d_k - min_d)/tau))
    
    # Let's compute the gradient of each distance term.
    
    # Wall terms
    for i in range(n):
        # x component
        if wall_dists_x[i] < wall_dists_y[i]:
            # wall_dist = 0.5 - |x_i|
            # d/dx_i = -sign(x_i)
            grad[2*i] += -np.sign(centers[i, 0])
        else:
            # wall_dist = 0.5 - |y_i|
            # d/dx_i = 0
            pass
        
        # y component
        if wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += -np.sign(centers[i, 1])
        else:
            pass
    
    # Pairwise terms
    # This is complex to vectorize perfectly for the gradient without loops or heavy memory.
    # Let's use a simpler approach: return None to let scipy use numerical gradient,
    # or implement a loop-based gradient.
    
    # For N=50, a loop-based gradient might be acceptable if it's not called too often.
    # But L-BFGS-B calls it many times.
    # Let's try to implement the pairwise part.
    
    # We need d(soft_min)/d(x_k) = sum_j w_j * d(d_j)/d(x_k)
    # d_j is the distance for pair (i,j) or wall i.
    
    # This is getting complex. Let's just use the objective function and let scipy do numerical gradient.
    # It's slower but more robust to implement correctly.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Simulated Annealing
    remaining = deadline - time.time()
    sa_budget = min(90.0, remaining * 0.7)
    best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
    
    # Phase 2: Soft-min optimization with L-BFGS-B
    remaining = deadline - time.time()
    if remaining > 10:
        x0 = best_centers.flatten()
        
        # Use a small tau to approximate the min closely
        tau = 0.005
        
        def obj(x):
            return _soft_min_obj(x, tau)
        
        try:
            result = opt.minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < obj(x0):
                polished = result.x.reshape(-1, 2)
                polished = np.clip(polished, -0.5, 0.5)
                polished_r = _compute_radius(polished)
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
        except Exception:
            pass
    
    # Phase 3: Final local SA polish
    remaining = deadline - time.time()
    if remaining > 5:
        best_centers, best_radius = _sa_search(best_centers, rng, remaining - 1)
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
