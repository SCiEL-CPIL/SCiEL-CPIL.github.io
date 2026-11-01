import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances
    # Using broadcasting for simplicity, N=50 is small enough
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, tau):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    Returns (value, gradient)
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraint distances into a flat array
    # We need to be careful with memory and indexing
    # Number of pairs: N*(N-1)/2
    # Number of walls: N
    iu, ju = np.triu_indices(N, k=1)
    n_pairs = len(iu)
    
    all_dists = np.empty(n_pairs + N)
    all_dists[:n_pairs] = pair_dists[iu, ju]
    all_dists[n_pairs:] = walls
    
    # Soft min calculation
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    # Avoid overflow in exp
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    soft_min = min_d - tau * log_sum
    
    value = -soft_min
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # 1. Wall contributions
    # D_wall_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # grad(D_wall_i) w.r.t x_i:
    # if 0.5 - |x_i| < 0.5 - |y_i|: -sign(x_i)
    # if 0.5 - |y_i| < 0.5 - |x_i|: -sign(y_i)
    # We want -sum(w * grad D).
    # -w * (-sign(x)) = w * sign(x)
    
    for i in range(N):
        w_wall = weights[n_pairs + i]
        if w_wall < 1e-12: 
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            # Both are limiting
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])

    # 2. Pair contributions
    # D_pair_ij = dist(i, j) / 2
    # grad(D_pair_ij) w.r.t c_i = (c_i - c_j) / (2 * dist(i,j))
    # We want -sum(w * grad D).
    # -w * (c_i - c_j) / (2d)
    
    for k in range(n_pairs):
        w_pair = weights[k]
        if w_pair < 1e-12:
            continue
        i = iu[k]
        j = ju[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        
        diff_ij = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
        
    return value, grad

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

def _random_initial(rng):
    """Generate a random initial configuration."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _lbfgs_optimize(x0, tau, time_budget):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f, g = _soft_min_obj_and_grad(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Multi-start with moderate tau
    # Try a few different initializations and tau values
    inits = [
        _initial_grid(),
        _random_initial(rng),
        _random_initial(rng)
    ]
    
    taus_phase1 = [0.01, 0.005]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        current = init.copy()
        for tau in taus_phase1:
            if time.time() > deadline - 25:
                break
            trial = _lbfgs_optimize(current.flatten(), tau, time_budget=10.0)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
            current = trial # Use the result as the next start point for smaller tau or next init
    
    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)
        
    # Phase 2: High precision polish with smaller tau
    if time.time() < deadline - 15:
        taus_phase2 = [0.001, 0.0005, 0.0001]
        current = best_centers.copy()
        for tau in taus_phase2:
            if time.time() > deadline - 10:
                break
            trial = _lbfgs_optimize(current.flatten(), tau, time_budget=10.0)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
            current = trial
            
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
