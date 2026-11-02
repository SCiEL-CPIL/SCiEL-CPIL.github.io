import numpy as np
from scipy.optimize import minimize
import time

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
    
    # Pairwise distances (Full N x N for accuracy)
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
    # For each i: if wall_x < wall_y: grad_x += sig * sign(x), grad_y += 0
    #             if wall_y < wall_x: grad_x += 0, grad_y += sig * sign(y)
    #             else: split
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    grad[2*np.nonzero(wx)[0]] += sig_wall[wx] * np.sign(centers[wx, 0])
    grad[2*np.nonzero(wy)[0]+1] += sig_wall[wy] * np.sign(centers[wy, 1])
    if np.any(wt):
        idx_wt = np.nonzero(wt)[0]
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])
    
    # Pair gradients (vectorized)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj  # shape (_NPAIRS, 2)
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    # factor_k = sig_pair[k] / dists[k], shape (_NPAIRS,)
    factor = sig_pair / dists_safe
    # contribution per pair: factor * diff_vec  shape (_NPAIRS, 2)
    contrib = factor[:, None] * diff_vec
    
    # Scatter: for pair (i,j), grad_i -= contrib, grad_j += contrib
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_width, s_height)
        
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()
    
    return best_grid if best_grid is not None else _initial_grid()

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Uses vectorized radius computation.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    # Dynamic step count based on budget
    est_steps = int(time_budget / 0.0005) 
    n_steps = max(10000, min(50000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Quick vectorized radius check
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
            if np.random.rand() < np.exp(delta_r / T):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _polish(centers, r_target, time_budget):
    """
    Polish a configuration using L-BFGS-B to maximize the minimum radius.
    Uses the soft violation objective.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    
    # If current is already good, try to push it further
    # We want to find a configuration where min_dist >= 2*r_target and walls >= r_target
    # If current_r is close to r_target, we can use r_target as the target.
    # If current_r < r_target, the violation will be high, and the optimizer will try to fix it.
    
    beta = 50.0
    # Increase beta gradually? Or keep fixed?
    # Fixed is simpler.
    
    v0 = current.flatten()
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    # Objective function
    def obj(v):
        return _soft_violation_obj(v, r_target, beta)
    
    def grad(v):
        return _soft_violation_grad(v, r_target, beta)
    
    try:
        # We want to minimize the violation.
        # If we can get violation to 0, it means we found a valid packing with radius >= r_target.
        # We don't know if r_target is achievable, but if we start from a good config, it's likely.
        # Actually, we want to MAXIMIZE the radius.
        # The soft violation approach is usually used for a FIXED target radius.
        # To maximize, we can:
        # 1. Start with r_target = current_r.
        # 2. Optimize to minimize violation.
        # 3. If violation is near 0, increase r_target slightly and repeat.
        # 4. Or, just use a very high r_target (like 0.1) and let the optimizer push as far as it can.
        #    But if r_target is too high, the gradient might be dominated by the infeasibility.
        
        # Better strategy:
        # Start with r_target = current_r.
        # Optimize.
        # Check new radius.
        # If improved, update r_target and continue.
        
        best_pos = current.copy()
        best_r = current_r
        
        # Iterative polishing
        # We have time_budget.
        # Each L-BFGS run might take ~1-5s.
        
        for _ in range(10): # Max 10 iterations
            if time.time() - start_time > time_budget:
                break
            
            r_target = best_r + 1e-5 # Slightly higher target to encourage improvement
            
            # Optimization
            res = minimize(
                obj,
                v0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            new_pos = res.x.reshape(N, 2)
            new_r = _compute_radius(new_pos)
            
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos.copy()
                v0 = new_pos.flatten()
            else:
                # If no improvement, break or continue with same target?
                # Usually if no improvement, the target was too aggressive or we hit a local min.
                # Try a slightly lower target? Or break?
                # Let's break if no significant improvement.
                if new_r < best_r - 1e-6:
                    break
    
    except Exception:
        pass
        
    return best_pos, best_r

def construct():
    start_time = time.time()
    
    best_pos = None
    best_r = -np.inf
    
    # 1. Generate initial configurations
    inits = [
        _initial_grid(),
        _hex_init()
    ]
    
    # Add random initializations
    rng = np.random.default_rng(0)
    for _ in range(3):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 2. Run Simulated Annealing for each init
    # Split time budget among inits and polishing
    # SA is good for global search.
    # Let's give 60% of time to SA, 40% to Polishing.
    total_budget = DEADLINE - (time.time() - start_time)
    sa_budget = total_budget * 0.6 / len(inits)
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 5: # Leave 5s for final polish
            break
            
        # Run SA
        sa_pos, sa_r = _simulated_annealing(init, rng, sa_budget)
        
        if sa_r > best_r:
            best_r = sa_r
            best_pos = sa_pos.copy()
            
        # Check time
        if time.time() > DEADLINE - 10:
            break
            
    # 3. Polish the best result found
    if best_pos is not None:
        polish_budget = DEADLINE - (time.time() - start_time) - 2 # Leave 2s safety
        if polish_budget > 5:
            polished_pos, polished_r = _polish(best_pos, best_r, polish_budget)
            if polished_r > best_r:
                best_r = polished_r
                best_pos = polished_pos.copy()
    
    if best_pos is None:
        # Fallback
        best_pos = inits[0]
        
    return best_pos
