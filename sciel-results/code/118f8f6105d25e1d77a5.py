"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with high-precision L-BFGS polishing
using a full-pairwise smooth violation objective.
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
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    if len(idx_wx) > 0:
        grad[2*idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
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
    Uses vectorized radius computation (no per-step KDTree build).
    Optimized move size and acceptance criteria.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    # Dynamic step count based on budget
    est_steps = int(time_budget / 0.0004)
    n_steps = max(15000, min(60000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    # Precompute upper triangle for fast pairwise min
    iu = _IU
    n_pairs = _NPAIRS
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Quick vectorized radius check using precomputed indices
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        # Vectorized pairwise min distance using precomputed indices
        ci = current[iu[0]]
        cj = current[iu[1]]
        diff = ci - cj
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)
        
        # Move generation - try multiple move strategies
        i = rng.integers(0, N)
        # Scale move size with temperature, but cap it
        move_size = 0.002 + 0.008 * (T / T_start)
        
        # Occasionally do a "kick" move (larger perturbation)
        if rng.random() < 0.05:
            move_size *= 3.0
        
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])
            
        trial = current.copy()
        trial[i] += direction * move_size
        trial = np.clip(trial, -0.5, 0.5)
        
        # Compute trial radius using same fast method
        trial_wall_x = 0.5 - np.abs(trial[:, 0])
        trial_wall_y = 0.5 - np.abs(trial[:, 1])
        trial_min_wall = np.min(np.minimum(trial_wall_x, trial_wall_y))
        
        tci = trial[iu[0]]
        tcj = trial[iu[1]]
        tdiff = tci - tcj
        tdist_sq = np.einsum('ij,ij->i', tdiff, tdiff)
        trial_min_pair = np.sqrt(np.min(tdist_sq)) / 2.0
        trial_r = min(trial_min_pair, trial_min_wall)
        
        delta_r = trial_r - curr_r
        
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    """
    Run L-BFGS-B to minimize violation for a specific target radius r_target.
    Uses vectorized objective and gradient.
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
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 10000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    """
    Binary search for the maximum feasible radius using L-BFGS polishing.
    Increased iterations and tighter tolerance.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Initial bounds
    lo = best_r
    hi = min(0.08, best_r * 1.05)
    
    search_steps = 15
    for i in range(search_steps):
        if time.time() - start_time > time_budget:
            break
            
        mid = (lo + hi) / 2
        beta = 200.0
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(15.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-7:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
            
    return best, best_r

def _epigraph_polish(centers, time_budget):
    """
    Epigraph formulation: minimize -t subject to constraints.
    Uses L-BFGS with augmented penalty for smooth constraints.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Start with current best radius and try to increase it
    r_current = best_r
    
    for iteration in range(8):
        if time.time() - start_time > time_budget:
            break
        
        # Target slightly above current
        r_target = r_current * 1.003
        if r_target > 0.08:
            r_target = 0.08
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(12.0, remaining * 0.8)
        
        # Use high beta for tight constraint enforcement
        beta = 2000 + iteration * 1000
        
        trial = _lbfgs_feasibility_polish(best.flatten(), r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > r_current:
            r_current = trial_r
            best = trial
    
    return best, r_current

def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using increasing beta and epigraph approach.
    Includes a "push" phase targeting slightly above current best.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Progressive sharpening of the violation objective
    betas = [500, 1000, 2000, 4000, 8000]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        r_target = best_r
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(15.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    # Epigraph polish phase
    if time.time() - start_time < time_budget - 8:
        epigraph_budget = min(20.0, time_budget - (time.time() - start_time) - 5)
        if epigraph_budget > 5:
            best, best_r = _epigraph_polish(best, epigraph_budget)
    
    # Push phase: try to target slightly above best_r
    if time.time() - start_time < time_budget - 5:
        for push_frac in [1.002, 1.005, 1.01, 1.015]:
            if time.time() - start_time > time_budget - 3:
                break
            r_push = best_r * push_frac
            if r_push > 0.08:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 3:
                break
            lbfgs_budget = min(10.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), r_push, 1500.0, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(777)
    
    # 1. Initialization - try multiple starting points
    inits = [
        _initial_grid(),
        _hex_init(),
    ]
    
    # Add a random initialization
    rng_init = np.random.default_rng(999)
    random_init = rng_init.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init)
    
    best_centers = None
    best_radius = 0
    
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)
        
    # 2. Simulated Annealing (Global Search) - multi-restart
    sa_budget_total = min(50.0, (deadline - time.time()) * 0.40)
    
    if sa_budget_total > 10:
        # First SA run
        sa1_budget = min(25.0, sa_budget_total * 0.5)
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa1_budget)
        
        # Second SA run with different seed for diversity
        if time.time() < deadline - 30:
            rng2 = np.random.default_rng(1234)
            # Perturb the best solution slightly for second SA
            perturbed = best_centers + rng2.normal(0, 0.01, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            sa2_budget = min(25.0, (deadline - time.time()) - 35)
            if sa2_budget > 5:
                c2, r2 = _simulated_annealing(perturbed, rng2, sa2_budget)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = c2
        
    # 3. Feasibility-Driven Radius Refinement (Coarse to Fine)
    if time.time() < deadline - 30:
        best_centers, best_radius = _refine_radius(
            best_centers, time_budget=min(40.0, deadline - time.time() - 15)
        )
        
    # 4. High Precision Polish (Final Tightening + Push)
    if time.time() < deadline - 8:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(25.0, deadline - time.time() - 3)
        )
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
