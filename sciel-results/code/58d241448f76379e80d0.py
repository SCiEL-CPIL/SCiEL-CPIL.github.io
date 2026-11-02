"""
Optimized packing of 50 equal circles in a unit square.
Combines vectorized Simulated Annealing for global search with a binary search 
strategy using high-precision L-BFGS polishing of a soft-violation objective.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances (Full N x N for accuracy, vectorized)
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Violation for pair (i,j): softplus(beta * (2*r_target - d_ij))
    Violation for wall i: softplus(beta * (r_target - w_i))
    Vectorized over pairs.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    # Pairwise distances (vectorized via upper-triangle indices)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
    # Softplus: log(1 + exp(z)) / beta
    # Clip to avoid overflow in exp
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)
    
    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta
    
    return float(np.sum(soft_wall) + np.sum(soft_pair))

def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective. Fully vectorized.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
    
    # Vectorized wall gradient contribution
    # If wall_x < wall_y: constraint is active on x. Grad w.r.t x is sig * sign(x)
    # If wall_y < wall_x: constraint is active on y. Grad w.r.t y is sig * sign(y)
    # If equal: split gradient
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    grad[2*idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    grad[2*idx_wy + 1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    
    if np.any(wt):
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt + 1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])
    
    # Pair gradients (vectorized)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj  # shape (_NPAIRS, 2)
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    # factor_k = sig_pair[k] / dists[k]
    factor = sig_pair / dists_safe
    # contribution per pair: factor * diff_vec
    contrib = factor[:, None] * diff_vec
    
    # Scatter: for pair (i,j), grad_i -= contrib, grad_j += contrib
    # Note: diff_vec = c_i - c_j. 
    # d/dc_i (c_i - c_j)^2 = 2(c_i - c_j). So grad_i gets + (c_i - c_j) * factor
    # d/dc_j (c_i - c_j)^2 = -2(c_i - c_j). So grad_j gets - (c_i - c_j) * factor
    # Wait, the objective is softplus(beta * (2r - d)). 
    # d/dx softplus = sig. 
    # d/dx (2r - d) = -d/dx d.
    # d/dc_i d = (c_i - c_j)/d. So d/dc_i (2r-d) = -(c_i-c_j)/d.
    # So grad_i += sig * (-(c_i-c_j)/d) = -sig/d * (c_i-c_j).
    # My code above: contrib = factor * diff_vec = (sig/d) * (c_i - c_j).
    # So grad_i should be -contrib, grad_j should be +contrib.
    
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _initial_grid():
    """Create initial hexagonal grid layout."""
    # Hexagonal packing: rows offset by half spacing
    # For 50 circles, use 7 rows: 7,8,7,8,7,8,5 = 50
    rows = [7, 8, 7, 8, 7, 8, 5]
    spacing_x = 1.0 / 8.5
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    centers = []
    y_pos = -0.5 + spacing_y * 0.5
    for r_idx, n_cols in enumerate(rows):
        offset = 0.5 * spacing_x if r_idx % 2 == 1 else 0.0
        x_start = -0.5 + spacing_x * 0.5 + offset
        for c in range(n_cols):
            x = x_start + c * spacing_x
            y = y_pos + r_idx * spacing_y
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    # Add small jitter to break symmetry
    rng = np.random.default_rng(12345)
    centers = centers + rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _simulated_annealing(centers, rng, time_budget):
    """
    Vectorized Simulated Annealing to maximize the minimum radius.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    
    # Estimate steps based on budget, assuming ~0.0003s per step
    est_steps = int(time_budget / 0.0003)
    n_steps = max(10000, min(80000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Quick vectorized radius check for current state
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        d = current[:, None, :] - current[None, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)
        
        # Move generation
        i = rng.integers(0, N)
        move_size = 0.002 + 0.008 * (T / T_start)
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])
            
        trial = current.copy()
        trial[i] += direction * move_size
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            # Accept with probability exp(-delta_r / T)
            if rng.random() < np.exp(-delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
        
    return best, best_r

def _lbfgs_polish(x0, r_target, beta, time_budget):
    """
    Run L-BFGS-B to minimize soft violation for a specific target radius.
    Returns the polished centers and the final violation.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 1000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        final_viol = _soft_violation_obj(polished.flatten(), r_target, beta)
        return polished, final_viol
    except Exception:
        return x0.reshape(-1, 2), np.inf

def _binary_search_polish(centers, r_start, time_budget):
    """
    Binary search for the maximum feasible radius using L-BFGS polishing.
    """
    start_time = time.time()
    current = centers.copy()
    best_r = r_start
    
    # We will try to increase the radius.
    # Start with a small increment, then binary search.
    # Actually, a more robust approach:
    # 1. Try to polish at r_start * 1.001, 1.002, etc? No, binary search is better.
    
    # Let's do a few iterations of "try to push radius up"
    # We maintain a lower bound lo (known feasible) and upper bound hi (known infeasible or high).
    
    lo = r_start
    # Estimate a high upper bound. The theoretical max for 50 circles is ~0.071377.
    # Let's set hi to lo * 1.05 or a known theoretical limit if we have it.
    hi = min(0.075, r_start * 1.1)
    
    # First, ensure we have a valid starting point by polishing at lo with high beta
    # This "tightens" the configuration.
    beta_start = 100.0
    
    # Phase 1: Tighten at current best radius
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        polished, viol = _lbfgs_polish(current.flatten(), lo, beta_start, remaining)
        if viol < 1e-6: # If we can satisfy constraints at lo with high beta, we are tight
            current = polished
            # Recalculate true radius, might have improved slightly
            r_check = _compute_radius(current)
            if r_check > lo:
                lo = r_check
                best_r = lo
    
    # Phase 2: Binary Search
    # We want to find the max r such that violation(r) is small.
    # We will iterate: mid = (lo+hi)/2. Polish at mid. If violation < threshold, lo=mid, else hi=mid.
    
    n_iter = 20 # Max binary search steps
    for _ in range(n_iter):
        if time.time() - start_time > time_budget - 2:
            break
        if hi - lo < 1e-10:
            break
            
        mid = (lo + hi) / 2
        
        # Use a high beta to make the constraint "hard"
        beta = 500.0
        
        remaining = time_budget - (time.time() - start_time) - 1
        if remaining < 2:
            break
            
        polished, viol = _lbfgs_polish(current.flatten(), mid, beta, remaining)
        
        # Check if the polished configuration is actually feasible for radius mid
        # The soft violation being small is a good indicator, but let's verify with true radius
        # If true radius >= mid - small_epsilon, we consider it feasible.
        true_r = _compute_radius(polished)
        
        if true_r >= mid - 1e-9:
            lo = mid
            current = polished
            best_r = true_r
        else:
            hi = mid
            
    return current, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # 1. Initial Configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # 2. Simulated Annealing (Global Search)
    # Allocate ~40% of time to SA
    sa_budget = min(40.0, (deadline - time.time()) * 0.4)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
        
    # 3. Initial Polish at current best radius
    # Allocate ~20% of time to initial tightening
    polish_budget = min(20.0, (deadline - time.time()) * 0.2)
    if polish_budget > 5:
        # Use a moderate beta to get a good tight configuration
        polished, viol = _lbfgs_polish(best_centers.flatten(), best_radius, beta=100.0, time_budget=polish_budget)
        r_check = _compute_radius(polished)
        if r_check > best_radius:
            best_radius = r_check
            best_centers = polished
    
    # 4. Binary Search Polish (High Precision)
    # Allocate remaining time
    remaining = deadline - time.time() - 2
    if remaining > 10:
        best_centers, best_radius = _binary_search_polish(best_centers, best_radius, remaining)
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers

if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r}")
    print(f"Score: {r / 0.071377103865}")
