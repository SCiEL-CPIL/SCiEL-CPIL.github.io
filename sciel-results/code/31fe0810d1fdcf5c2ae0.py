"""
Optimized packing of 50 equal circles in a unit square.
Combines SA for global search with vectorized soft-violation L-BFGS polishing.
Uses upper-triangle index precomputation and fully vectorized gradients.
Incorporates Delaunay-inspired contact-graph local search and multi-start restarts.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 147.0  # Safety margin before 150s kill

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
    
    # Pairwise distances using KDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Uses log-sum-exp based soft max of violations.
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
        grad[2*idx_wx] += sig_wall[idx_wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += sig_wall[idx_wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] += 0.5 * sig_wall[idx_wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[idx_wt] * np.sign(centers[idx_wt, 1])
    
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
    Soft-min objective: minimize negative soft-min of all constraint distances.
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
    
    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -50, 50)
    z_max = np.max(z)
    logsumexp = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min_val = -logsumexp / beta
    
    return float(-soft_min_val)

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
    z_max = np.max(z)
    z_shifted = z - z_max
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp
    
    # Wall gradients
    n_walls = N
    w_weights = weights[:n_walls]
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
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

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    rng = np.random.default_rng(42)
    
    for attempt in range(12):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        s_width = 1.0 / (cols + 0.3)
        s_height = 1.0 / (rows * 0.866 + 0.3)
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

def _random_init(rng):
    """Create random initial layout."""
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers

def _perturbed_init(base, rng, scale=0.02):
    """Create perturbed version of a base layout."""
    centers = base + rng.normal(0, scale, base.shape)
    return np.clip(centers, -0.49, 0.49)

def _contact_graph_local_search(centers, rng, time_budget, r_target=None):
    """
    Local search that identifies near-contact pairs (Delaunay-like edges)
    and performs coordinated moves to improve the minimum radius.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    if r_target is None:
        r_target = best_r * 1.001
    
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Build contact graph: pairs within 2*r_target * 1.15
        tree = cKDTree(current)
        pairs = tree.query_pairs(2 * r_target * 1.15)
        
        if len(pairs) == 0:
            break
        
        # Weight edges by how close they are to the constraint
        edge_weights = []
        for i, j in pairs:
            d = np.linalg.norm(current[i] - current[j])
            w = np.exp(-(2 * r_target - d) / 0.003)
            edge_weights.append(w)
        
        edge_weights = np.array(edge_weights)
        total_w = np.sum(edge_weights)
        if total_w < 1e-10:
            break
        
        # Pick a random edge from the contact graph
        probs = edge_weights / total_w
        edge_idx = rng.choice(len(pairs), p=probs)
        i, j = pairs[edge_idx]
        
        # Also consider wall constraints
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        # Decide move strategy
        strategy = rng.random()
        move_size = 0.002 + 0.008 * (1.0 - it / max_iters)
        
        trial = current.copy()
        
        if strategy < 0.4:
            # Move pair apart
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            trial[i] += unit * move_size * rng.uniform(0.3, 0.7)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.7)
        elif strategy < 0.6:
            # Move single circle away from nearest neighbor
            dists, idx = tree.query(current[i], k=2)
            nn = idx[1]
            diff = current[i] - current[nn]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            trial[i] += unit * move_size
        elif strategy < 0.8:
            # Move toward wall if near wall
            wx = wall_x[i]
            wy = wall_y[i]
            if wx < wy:
                trial[i, 0] += move_size * np.sign(current[i, 0]) * -1
            else:
                trial[i, 1] += move_size * np.sign(current[i, 1]) * -1
        else:
            # Random small move
            trial[i] += rng.normal(0, move_size, 2)
        
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        curr_r = _compute_radius(current)
        
        if trial_r >= curr_r:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            # Accept with probability based on improvement deficit
            delta = trial_r - curr_r
            if rng.random() < np.exp(delta / 0.001):
                current = trial
    
    return best, best_r

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Build tree once per step
        tree = cKDTree(current)
        dists, idx = tree.query(current, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)
        
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        idx_sel = rng.choice(len(all_dists), p=weights)
        
        if idx_sel < N:
            i = idx_sel
            move_size = 0.003 + 0.012 * T / T_start
            direction = rng.integers(0, 4)
            delta = np.zeros(2)
            if direction == 0:
                delta[0] = move_size * rng.uniform(0.5, 1.0)
            elif direction == 1:
                delta[0] = -move_size * rng.uniform(0.5, 1.0)
            elif direction == 2:
                delta[1] = move_size * rng.uniform(0.5, 1.0)
            else:
                delta[1] = -move_size * rng.uniform(0.5, 1.0)
            
            trial = current.copy()
            trial[i] += delta
            trial = np.clip(trial, -0.5, 0.5)
        else:
            pair_idx = idx_sel - N
            i = pair_idx
            j = idx[i, 1]
            
            move_size = 0.003 + 0.012 * T / T_start
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.3, 0.7)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.7)
            trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
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

def _lbfgs_soft_min(x0, beta, time_budget=15.0):
    """Run L-BFGS-B optimization on the soft-min objective."""
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
            options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14,
                     'maxfun': 3000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_soft_violation(x0, r_target, beta, time_budget=10.0):
    """Run L-BFGS-B optimization on the soft-violation objective for a target radius."""
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
            options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-14,
                     'maxfun': 2500, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using L-BFGS with decreasing beta on soft-min objective.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    for beta in [100, 200, 400, 800, 1500]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = best.flatten().copy()
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3:
            break
        
        trial = _lbfgs_soft_min(x0, beta=beta, time_budget=min(12, remaining))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
    
    return best, best_r

def _binary_search_with_violation(centers, lo, hi, time_budget):
    """
    Binary search for maximum feasible radius using soft-violation L-BFGS.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(8):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        x0 = best.flatten().copy()
        trial = _lbfgs_soft_violation(x0, mid, beta=500, time_budget=min(8, remaining - 2))
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            if trial_r > best_r:
                best_r = trial_r
                best = trial
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try multiple initializations
    rng = np.random.default_rng(777)
    
    inits = [
        _initial_grid(),
        _hex_init(),
        _random_init(rng),
    ]
    
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
    
    # Phase 1: Simulated Annealing
    sa_budget = min(25.0, (deadline - time.time()) * 0.20)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 1.5: Contact-graph local search (Delaunay-inspired)
    if time.time() < deadline - 80:
        cg_budget = min(10.0, deadline - time.time() - 70)
        if cg_budget > 3:
            trial, trial_r = _contact_graph_local_search(best_centers, rng, cg_budget, r_target=best_radius)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
    
    # Phase 2: L-BFGS soft-min optimization with increasing beta
    if time.time() < deadline - 50:
        x0 = best_centers.flatten()
        for beta in [50, 100, 200, 400]:
            if time.time() > deadline - 40:
                break
            remaining = deadline - time.time() - 15
            if remaining < 5:
                break
            trial = _lbfgs_soft_min(x0, beta=beta, time_budget=min(10, remaining))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 2.5: Another round of contact-graph local search after polishing
    if time.time() < deadline - 35:
        cg_budget = min(8.0, deadline - time.time() - 28)
        if cg_budget > 3:
            trial, trial_r = _contact_graph_local_search(best_centers, rng, cg_budget, r_target=best_radius)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
    
    # Phase 3: High-precision polish with very high beta
    if time.time() < deadline - 25:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(15, deadline - time.time() - 10)
        )
    
    # Phase 4: Binary search with soft-violation feasibility
    if time.time() < deadline - 10:
        lo = best_radius
        hi = min(0.075, best_radius * 1.05)
        best_centers, best_radius = _binary_search_with_violation(
            best_centers, lo, hi, time_budget=min(12, deadline - time.time() - 4)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
