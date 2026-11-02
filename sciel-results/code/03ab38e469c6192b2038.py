"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with high-precision L-BFGS polishing
using a full-pairwise smooth violation objective, plus a soft-min epigraph approach
for final polishing.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 146.0  # Safety margin before 150s kill

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
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
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
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
    
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
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    factor = sig_pair / dists_safe
    contrib = factor[:, None] * diff_vec
    
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _soft_min_obj(x, beta):
    """
    Epigraph-style soft-min objective: minimize the soft-min of all constraints.
    The constraints are: pairwise distances >= 2r and wall distances >= r.
    We use a fixed r_target that we try to maximize via outer loop.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0
    
    # All constraints values (we want them all >= 0, i.e. r <= min)
    # Soft-min of all values: -logsumexp(-beta * values) / beta
    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -50, 50)
    logsumexp = np.max(z) + np.log(np.sum(np.exp(z - np.max(z))))
    soft_min_val = -logsumexp / beta
    
    return float(-soft_min_val)  # minimize negative soft-min

def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0
    
    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -50, 50)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp  # softmax weights
    
    # Gradient of -soft_min = -(-1/beta) * d(logsumexp)/dx = (1/beta) * sum_k weights_k * d(-val_k)/dx
    # = -(1/beta) * sum_k weights_k * d(val_k)/dx
    # Wait: soft_min = -logsumexp(-beta*vals)/beta
    # d(soft_min)/dx = -1/beta * d(logsumexp(-beta*vals))/dx
    # = -1/beta * sum_k exp(-beta*val_k)/sum_j exp(-beta*val_j) * (-beta) * d(val_k)/dx
    # = sum_k weights_k * d(val_k)/dx
    # We want to minimize -soft_min, so gradient is -sum_k weights_k * d(val_k)/dx
    
    # Wall gradients
    n_walls = N
    w_weights = weights[:n_walls]
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    # d(wall_i)/d(x_i) = -sign(x_i) if x is limiting, -sign(y_i) if y is limiting
    if len(idx_wx) > 0:
        grad[2*idx_wx] -= w_weights[idx_wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] -= w_weights[idx_wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 1])
    
    # Pair gradients
    p_weights = weights[n_walls:]
    dists_safe = np.maximum(dists, 1e-12)
    # d(half_dist_ij)/d(c_i) = -(c_i - c_j) / dist_ij
    # d(half_dist_ij)/d(c_j) = (c_i - c_j) / dist_ij
    factor = p_weights / dists_safe
    contrib = factor[:, None] * diff
    
    np.add.at(grad, 2 * _IU[0], contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, -contrib[:, 1])
    
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

def _deformation_init():
    """
    Create initial layout using a smooth deformation field applied to a hexagonal grid.
    The deformation is parameterized by low-frequency Fourier modes, ensuring
    local invertibility and producing dense, non-overlapping configurations.
    """
    # Base hexagonal grid
    m = int(np.ceil(np.sqrt(N)))
    rows = m
    cols = m
    
    # Generate base hexagonal positions
    base = []
    s = 1.0 / (m + 0.3)
    for j in range(rows):
        for i in range(cols):
            x = (i - (cols - 1) / 2) * s
            y = (j - (rows - 1) / 2) * s * 0.866
            if j % 2 == 1:
                x += s / 2
            base.append([x, y])
    base = np.array(base[:N])
    
    # Apply a smooth deformation field using low-frequency modes
    rng = np.random.default_rng(42)
    n_modes = 3
    # Fourier coefficients for deformation
    freqs_x = rng.uniform(1, 4, n_modes)
    freqs_y = rng.uniform(1, 4, n_modes)
    amps_x = rng.uniform(0.01, 0.04, n_modes)
    amps_y = rng.uniform(0.01, 0.04, n_modes)
    phases_x = rng.uniform(0, 2 * np.pi, n_modes)
    phases_y = rng.uniform(0, 2 * np.pi, n_modes)
    
    deformed = base.copy()
    for k in range(n_modes):
        deformed[:, 0] += amps_x[k] * np.sin(freqs_x[k] * base[:, 0] + phases_x[k]) * np.cos(freqs_y[k] * base[:, 1] + phases_y[k])
        deformed[:, 1] += amps_y[k] * np.sin(freqs_y[k] * base[:, 1] + phases_y[k]) * np.cos(freqs_x[k] * base[:, 0] + phases_x[k])
    
    # Small random perturbation
    deformed += rng.normal(0, 0.003, deformed.shape)
    deformed = np.clip(deformed, -0.49, 0.49)
    
    return deformed

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    rng = np.random.default_rng(42)
    
    for attempt in range(8):
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
    Uses incremental distance updates for speed.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.012
    T_end = 0.00003
    est_steps = int(time_budget / 0.00035)
    n_steps = max(20000, min(80000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    iu = _IU
    iu0, iu1 = iu[0], iu[1]
    
    # Precompute which pairs involve each point for incremental updates
    # For point i, find all pairs where i appears as first or second element
    pairs_of = [[] for _ in range(N)]
    for p in range(_NPAIRS):
        pairs_of[iu0[p]].append(p)
        pairs_of[iu1[p]].append(p)
    
    # Current distances (cached)
    ci = current[iu0]
    cj = current[iu1]
    diff = ci - cj
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    
    # Current wall distances
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current radius from cached values
        min_wall = np.min(walls)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)
        
        i = rng.integers(0, N)
        # Adaptive move size: larger early, smaller late
        frac = T / T_start
        move_size = 0.003 + 0.007 * frac
        
        # Occasionally do larger jumps
        if rng.random() < 0.08:
            move_size *= 2.5
        
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])
            
        old_pos = current[i].copy()
        new_pos = old_pos + direction * move_size
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Incremental update: only recompute distances involving point i
        old_wall_x = 0.5 - abs(old_pos[0])
        old_wall_y = 0.5 - abs(old_pos[1])
        old_wall = min(old_wall_x, old_wall_y)
        
        new_wall_x = 0.5 - abs(new_pos[0])
        new_wall_y = 0.5 - abs(new_pos[1])
        new_wall = min(new_wall_x, new_wall_y)
        
        # Update wall array
        walls[i] = new_wall
        
        # Update pair distances involving point i
        for p in pairs_of[i]:
            if iu0[p] == i:
                j = iu1[p]
                d = new_pos - current[j]
            else:
                j = iu0[p]
                d = current[j] - new_pos
            dist_sq[p] = np.dot(d, d)
        
        # Compute trial radius
        trial_min_wall = np.min(walls)
        trial_min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        trial_r = min(trial_min_pair, trial_min_wall)
        
        delta_r = trial_r - curr_r
        
        if delta_r >= 0:
            current[i] = new_pos
            if trial_r > best_r:
                best_r = trial_r
                best = current.copy()
        else:
            # Accept with probability exp(delta_r / T)
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current[i] = new_pos
            else:
                # Revert: restore old wall and pair distances
                walls[i] = old_wall
                for p in pairs_of[i]:
                    if iu0[p] == i:
                        j = iu1[p]
                        d = old_pos - current[j]
                    else:
                        j = iu0[p]
                        d = current[j] - old_pos
                    dist_sq[p] = np.dot(d, d)
        
        T *= cooling
    
    return best, best_r

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    """
    Run L-BFGS-B to minimize violation for a specific target radius r_target.
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
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14,
                     'maxfun': 25000, 'disp': False, 'maxls': 50}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_soft_min_polish(x0, beta, time_budget):
    """
    Run L-BFGS-B to maximize soft-min radius.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_obj(x, beta)
        g = _soft_min_grad(x, beta)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13,
                     'maxfun': 15000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    """
    Binary search for the maximum feasible radius using L-BFGS polishing.
    Uses progressive beta increase for tighter convergence.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    lo = best_r
    hi = min(0.08, best_r * 1.05)
    
    search_steps = 22
    for i in range(search_steps):
        if time.time() - start_time > time_budget:
            break
            
        mid = (lo + hi) / 2
        # Progressive beta: start moderate, increase for later (tighter) searches
        progress = i / max(search_steps - 1, 1)
        beta = 200.0 + 800.0 * progress
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(18.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-8:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
            
    return best, best_r

def _coordinate_descent_polish(centers, time_budget):
    """
    Coordinate descent polish: iterate over each point and each coordinate,
    finding the best local move that increases the minimum radius.
    This is a simple but effective final refinement.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    iu0, iu1 = _IU[0], _IU[1]
    
    # Precompute pairs for each point
    pairs_of = [[] for _ in range(N)]
    for p in range(_NPAIRS):
        pairs_of[iu0[p]].append(p)
        pairs_of[iu1[p]].append(p)
    
    # Cached distances
    ci = current[iu0]
    cj = current[iu1]
    diff = ci - cj
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    max_iters = 200
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                old_val = current[i, coord]
                
                # Try small perturbations in both directions
                for delta in [0.0005, 0.001, 0.002, 0.005, -0.0005, -0.001, -0.002, -0.005]:
                    new_val = old_val + delta
                    if new_val < -0.5 or new_val > 0.5:
                        continue
                    
                    # Update wall distance for point i
                    if coord == 0:
                        new_wall_x = 0.5 - abs(new_val)
                        new_wall = min(new_wall_x, wall_y[i])
                    else:
                        new_wall_y = 0.5 - abs(new_val)
                        new_wall = min(wall_x[i], new_wall_y)
                    
                    # Update pair distances involving point i
                    old_walls_i = walls[i]
                    walls[i] = new_wall
                    
                    for p in pairs_of[i]:
                        if iu0[p] == i:
                            j = iu1[p]
                            d = current[i].copy()
                            d[coord] = new_val
                            d = d - current[j]
                        else:
                            j = iu0[p]
                            d = current[j].copy()
                            d[coord] = new_val
                            d = current[i] - d
                        dist_sq[p] = np.dot(d, d)
                    
                    # Compute trial radius
                    trial_min_wall = np.min(walls)
                    trial_min_pair = np.sqrt(np.min(dist_sq)) / 2.0
                    trial_r = min(trial_min_pair, trial_min_wall)
                    
                    if trial_r > best_r + 1e-12:
                        # Accept this move
                        current[i, coord] = new_val
                        best_r = trial_r
                        best = current.copy()
                        improved = True
                        break
                    else:
                        # Revert
                        walls[i] = old_walls_i
                        for p in pairs_of[i]:
                            if iu0[p] == i:
                                j = iu1[p]
                                d = current[i] - current[j]
                            else:
                                j = iu0[p]
                                d = current[j] - current[i]
                            dist_sq[p] = np.dot(d, d)
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using multiple strategies:
    1. Progressive sharpening with feasibility objective
    2. Soft-min epigraph polish
    3. Push phase targeting slightly above current best
    4. Final ultra-tight polish
    5. Coordinate descent micro-polish
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Phase 1: Progressive sharpening of violation objective
    betas = [500, 1000, 2000, 4000, 8000, 12000]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        r_target = best_r
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 4:
            break
        lbfgs_budget = min(14.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    # Phase 2: Soft-min epigraph polish
    if time.time() - start_time < time_budget - 8:
        sm_betas = [200, 500, 1000, 2000, 4000]
        for beta in sm_betas:
            if time.time() - start_time > time_budget - 4:
                break
            x0 = best.flatten()
            remaining = time_budget - (time.time() - start_time)
            if remaining < 4:
                break
            lbfgs_budget = min(14.0, remaining * 0.85)
            
            trial = _lbfgs_soft_min_polish(x0, beta, lbfgs_budget)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_r:
                best_r = trial_r
                best = trial
            
    # Phase 3: Push phase - target slightly above best_r
    if time.time() - start_time < time_budget - 4:
        for push_frac in [1.001, 1.003, 1.005, 1.008, 1.012, 1.018]:
            if time.time() - start_time > time_budget - 3:
                break
            r_push = best_r * push_frac
            if r_push > 0.08:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 3:
                break
            lbfgs_budget = min(12.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), r_push, 4000.0, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    # Phase 4: Ultra-tight final polish with very high beta
    if time.time() - start_time < time_budget - 2:
        for beta in [8000, 15000, 25000]:
            if time.time() - start_time > time_budget - 1.5:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 2:
                break
            lbfgs_budget = min(15.0, remaining * 0.9)
            trial = _lbfgs_feasibility_polish(best.flatten(), best_r, beta, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    # Phase 5: Coordinate descent micro-polish
    if time.time() - start_time < time_budget - 1:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            trial, trial_r = _coordinate_descent_polish(best, remaining * 0.8)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(777)
    
    # 1. Initialization - try multiple diverse starts
    inits = [
        _initial_grid(),
        _hex_init(),
        _deformation_init(),
    ]
    
    rng_init = np.random.default_rng(999)
    random_init = rng_init.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init)
    
    # Add a second random init with different seed
    rng_init2 = np.random.default_rng(4242)
    random_init2 = rng_init2.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init2)
    
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
        
    # 2. Simulated Annealing - multi-restart with tighter budget
    sa_budget_total = min(35.0, (deadline - time.time()) * 0.30)
    
    if sa_budget_total > 8:
        sa1_budget = min(20.0, sa_budget_total * 0.55)
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa1_budget)
        
        # Second SA restart from perturbed best
        if time.time() < deadline - 35:
            rng2 = np.random.default_rng(1234)
            perturbed = best_centers + rng2.normal(0, 0.008, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            sa2_budget = min(15.0, (deadline - time.time()) - 40)
            if sa2_budget > 4:
                c2, r2 = _simulated_annealing(perturbed, rng2, sa2_budget)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = c2
        
    # 3. Feasibility-Driven Radius Refinement
    if time.time() < deadline - 25:
        refine_budget = min(45.0, deadline - time.time() - 12)
        best_centers, best_radius = _refine_radius(
            best_centers, time_budget=refine_budget
        )
        
    # 4. High Precision Polish
    if time.time() < deadline - 6:
        polish_budget = min(30.0, deadline - time.time() - 2)
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=polish_budget
        )
        
    # 5. Final micro-polish if time remains
    if time.time() < deadline - 2:
        final_budget = min(15.0, deadline - time.time() - 1)
        if final_budget > 3:
            cur_r = _compute_radius(best_centers)
            trial = _lbfgs_feasibility_polish(
                best_centers.flatten(), cur_r, 20000.0, final_budget
            )
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
