"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with a Smoothed Max-Min objective 
polishing strategy using L-BFGS. This avoids the local minima traps of 
pure violation-minimization by directly optimizing a smooth surrogate for 
the max-min problem.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 148.0  # Safety margin before 150s kill

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

def _smooth_max_min_obj(x, alpha):
    """
    Objective: Minimize sum of (1/d)^alpha for pairs and walls.
    This is a smooth approximation of the max-min problem.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    # Ensure positive to avoid inf/nan, clip to small epsilon
    walls_safe = np.maximum(walls, 1e-6)
    
    # Pairwise distances
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists_sq = np.sum(diff ** 2, axis=1)
    dists = np.sqrt(dists_sq)
    dists_safe = np.maximum(dists, 1e-6)
    
    # Objective: sum (1/d)^alpha
    # Using log-sum-exp style or direct sum? Direct sum is fine for moderate alpha.
    # To avoid overflow for large alpha, we can scale, but let's stick to standard first.
    # Actually, for alpha > 20, (1/1e-6)^20 is huge. 
    # Better objective: -log(sum(exp(-alpha * d))) is the log-sum-exp approximation of min.
    # Let's use the LogSumExp approximation of the minimum distance.
    # min(d) ~ -(1/alpha) * log(sum(exp(-alpha * d)))
    # Minimizing min(d) is equivalent to maximizing min(d).
    # So we want to MAXIMIZE min(d).
    # Or MINIMIZE -min(d).
    # Let's minimize: (1/alpha) * log(sum(exp(-alpha * d)))
    # This is a smooth lower bound on the min distance. As alpha -> inf, it approaches min(d).
    
    # We want to MAXIMIZE the min distance.
    # So we MINIMIZE the negative of the smooth min.
    # Smooth Min = -(1/alpha) * log( sum( exp(-alpha * d) ) )
    # Objective to minimize: -Smooth Min = (1/alpha) * log( sum( exp(-alpha * d) ) )
    
    # Let's compute the log-sum-exp for pairs and walls separately then combine?
    # Or just sum all distances into one vector.
    
    all_dists = np.concatenate([dists, walls])
    
    # LogSumExp trick: max_val = max(-alpha * d)
    # log(sum(exp(z))) = max_z + log(sum(exp(z - max_z)))
    z = -alpha * all_dists
    max_z = np.max(z)
    if not np.isfinite(max_z):
        return 0.0
        
    log_sum_exp = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    # Objective: (1/alpha) * log_sum_exp
    # Wait, if d is large, z is very negative, exp(z) is 0.
    # If d is small (close), z is less negative.
    # The min d corresponds to the largest z (least negative).
    # So log_sum_exp is dominated by the smallest d.
    # (1/alpha) * log_sum_exp approximates -min_d.
    # We want to MAXIMIZE min_d, so MINIMIZE -min_d.
    # So objective = (1/alpha) * log_sum_exp.
    
    return float((1.0 / alpha) * log_sum_exp)

def _smooth_max_min_grad(x, alpha):
    """
    Gradient of the smooth max-min objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    walls_safe = np.maximum(walls, 1e-6)
    
    # Pairwise distances
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists_sq = np.sum(diff ** 2, axis=1)
    dists = np.sqrt(dists_sq)
    dists_safe = np.maximum(dists, 1e-6)
    
    all_dists = np.concatenate([dists, walls])
    
    # Compute weights for LogSumExp
    z = -alpha * all_dists
    max_z = np.max(z)
    if not np.isfinite(max_z):
        return grad
        
    exp_z = np.exp(z - max_z)
    sum_exp_z = np.sum(exp_z)
    
    # Probability weights: w_k = exp(z_k) / sum(exp(z_k))
    # Note: log_sum_exp = max_z + log(sum_exp_z)
    # Objective = (1/alpha) * (max_z + log(sum_exp_z))
    # dObj/dx = (1/alpha) * d(log(sum_exp_z))/dx  (max_z is constant w.r.t x in derivative sense? No, max_z depends on x)
    # Actually, the derivative of logsumexp(z(x)) is sum( (exp(z_k)/sum_exp_z) * dz_k/dx )
    # So dObj/dx = (1/alpha) * sum( w_k * dz_k/dx )
    # where w_k = exp(z_k) / sum_exp_z
    
    w = exp_z / sum_exp_z
    
    # Now distribute gradients
    # For pairs:
    # z_k = -alpha * d_k
    # dz_k/dx_i = -alpha * (d d_k/dx_i)
    # d_k = ||c_i - c_j||
    # d d_k/dx_i = (c_i - c_j) / d_k
    # So dz_k/dx_i = -alpha * (c_i - c_j) / d_k
    
    # Contribution to grad from pair k:
    # grad_i += w_k * (1/alpha) * dz_k/dx_i = w_k * (1/alpha) * (-alpha * (c_i - c_j) / d_k)
    # grad_i += -w_k * (c_i - c_j) / d_k
    # grad_j += -w_k * (c_j - c_i) / d_k = w_k * (c_i - c_j) / d_k
    
    # Wait, let's re-verify signs.
    # We are MINIMIZING Obj = (1/alpha) log(sum(exp(-alpha d)))
    # This approximates -min(d).
    # We want to MAXIMIZE min(d).
    # So minimizing Obj is correct.
    
    # Gradient of Obj w.r.t c_i:
    # d/d c_i [ (1/alpha) log(sum_k exp(-alpha d_k)) ]
    # = (1/alpha) * (1/sum) * sum_k exp(-alpha d_k) * (-alpha) * d d_k/d c_i
    # = - sum_k (exp(-alpha d_k)/sum) * d d_k/d c_i
    # = - sum_k w_k * (c_i - c_j) / d_k  (for k involving i)
    
    # So for pair (i,j) with weight w_k:
    # grad_i += -w_k * (c_i - c_j) / d_k
    # grad_j += -w_k * (c_j - c_i) / d_k = w_k * (c_i - c_j) / d_k
    
    # Let's check direction.
    # If c_i is close to c_j, (c_i - c_j)/d_k is unit vector from j to i.
    # grad_i points from j to i? No, -w_k * (c_i - c_j)/d_k points from i to j.
    # If we move c_i in direction of grad_i (descent), we move c_i towards j.
    # That INCREASES the distance? No, moving towards j DECREASES distance.
    # We want to INCREASE distance.
    # So we should move AGAINST the gradient?
    # L-BFGS minimizes the function.
    # If Obj increases when d decreases, then grad points in direction of decreasing d.
    # Moving along -grad (ascent of d, descent of Obj) should increase d.
    # Let's check:
    # Obj ~ -min_d.
    # d(Obj)/d c_i ~ - d(min_d)/d c_i.
    # If we move c_i to increase min_d, d(min_d)/d c_i > 0.
    # Then d(Obj)/d c_i < 0.
    # Gradient points in direction of decreasing Obj.
    # So moving along -grad (which is opposite to gradient) increases Obj?
    # No, gradient descent moves along -grad to minimize Obj.
    # If Obj is -min_d, minimizing Obj means making -min_d smaller, i.e., min_d larger.
    # So Gradient Descent on Obj should increase min_d.
    
    # Let's re-evaluate the sign in the formula.
    # grad_i = - sum_k w_k * (c_i - c_j) / d_k
    # Vector (c_i - c_j) points from j to i.
    # So - (c_i - c_j) points from i to j.
    # So grad_i points from i to j.
    # Gradient descent step: c_i_new = c_i - lr * grad_i.
    # c_i moves in direction -grad_i, which is from j to i.
    # Moving from j to i INCREASES distance.
    # Correct!
    
    # So:
    # contrib_i = -w_k * (c_i - c_j) / d_k
    # contrib_j = -w_k * (c_j - c_i) / d_k = w_k * (c_i - c_j) / d_k
    
    # Pair indices
    n_pairs = len(_IU[0])
    w_pairs = w[:n_pairs]
    d_pairs = dists_safe[:n_pairs]
    
    # Vectorized contribution
    # diff_vec = c_i - c_j
    # factor = w_pairs / d_pairs
    # contrib = factor[:, None] * diff_vec
    # grad_i -= contrib
    # grad_j += contrib
    
    diff_vec = ci - cj
    factor = w_pairs / d_pairs
    contrib = factor[:, None] * diff_vec
    
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    # Wall indices
    w_walls = w[n_pairs:]
    d_walls = walls_safe
    
    # Wall distance w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # d w_i / d x_i = -sign(x_i) if 0.5 - |x_i| is the min
    # d w_i / d y_i = -sign(y_i) if 0.5 - |y_i| is the min
    
    # z_wall = -alpha * w_i
    # dz_wall/dx_i = -alpha * d w_i/dx_i
    # grad_i += -w_wall_k * dz_wall/dx_i ?
    # No, the formula is:
    # grad_i += - sum_k w_k * (d z_k/dx_i) * (1/alpha) ?
    # Let's stick to the previous derivation:
    # dObj/dx = - sum_k w_k * (d d_k/dx)
    # For walls, d_k is w_i.
    # So grad_i += -w_wall * (d w_i/dx_i)
    
    # d w_i/dx_i:
    # If wall_x < wall_y: w_i = 0.5 - |x_i|. d/dx = -sign(x_i).
    # If wall_y < wall_x: w_i = 0.5 - |y_i|. d/dx = 0.
    # If equal: split.
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    # grad_x contribution: -w_wall * (-sign(x)) = w_wall * sign(x)
    # grad_y contribution: -w_wall * (-sign(y)) = w_wall * sign(y)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    if len(idx_wx) > 0:
        grad[2*idx_wx] += w_walls[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += w_walls[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] += 0.5 * w_walls[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * w_walls[wt] * np.sign(centers[idx_wt, 1])
        
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

def _random_init():
    """Create a random initial layout."""
    rng = np.random.default_rng(999)
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.02
    T_end = 0.0001
    n_steps = 40000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Quick vectorized radius check
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        # Vectorized pairwise min distance
        d = current[:, None, :] - current[None, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)
        
        # Move generation
        i = rng.integers(0, N)
        move_size = 0.005 + 0.01 * T / T_start
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
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _smooth_polish(centers, time_budget):
    """
    Polish using the Smoothed Max-Min objective.
    Iteratively increases alpha to sharpen the approximation of the min-distance.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Start with low alpha for smoothness, increase for accuracy
    alphas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
    
    for alpha in alphas:
        if time.time() - start_time > time_budget:
            break
            
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(10.0, remaining * 0.85)
        
        def obj_and_grad(x):
            f = _smooth_max_min_obj(x, alpha)
            g = _smooth_max_min_grad(x, alpha)
            return f, g
            
        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxfun': 5000, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            trial_r = _compute_radius(polished)
            
            if trial_r > best_r:
                best_r = trial_r
                best = polished
        except Exception:
            pass
            
    return best, best_r

def _local_search(centers, time_budget):
    """
    A simple local search that tries to push circles apart if they are too close.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    n_iters = 3000
    step_size = 0.01
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
            
        curr_r = _compute_radius(current)
        
        # Find the pair with minimum distance
        diff = current[:, None, :] - current[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        i, j = min_idx
        
        # Also check walls
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        walls = np.minimum(wall_x, wall_y)
        min_wall_idx = np.argmin(walls)
        
        min_pair_dist = np.sqrt(dist_sq[i, j])
        min_wall_dist = walls[min_wall_idx]
        
        # Try to move the most constrained circle
        if min_pair_dist / 2.0 < min_wall_dist:
            # Move i and j apart
            direction = current[i] - current[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                trial = current.copy()
                trial[i] += direction * step_size
                trial[j] -= direction * step_size
                trial = np.clip(trial, -0.5, 0.5)
                trial_r = _compute_radius(trial)
                if trial_r > best_r:
                    best_r = trial_r
                    best = trial.copy()
                    current = trial
        else:
            # Move the circle closest to the wall away from it
            idx = min_wall_idx
            x = current[idx, 0]
            y = current[idx, 1]
            trial = current.copy()
            if 0.5 - x < x + 0.5: # Closer to right
                trial[idx, 0] -= step_size
            else:
                trial[idx, 0] += step_size
            if 0.5 - y < y + 0.5: # Closer to top
                trial[idx, 1] -= step_size
            else:
                trial[idx, 1] += step_size
            trial = np.clip(trial, -0.5, 0.5)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
                current = trial
                
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    # 1. Initialization
    inits = [
        _initial_grid(),
        _hex_init(),
        _random_init(),
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
        
    # 2. Simulated Annealing (Global Search)
    sa_budget = min(40.0, (deadline - time.time()) * 0.3)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
        
    # 3. Local Search (to escape shallow local minima)
    if time.time() < deadline - 60:
        best_centers, best_radius = _local_search(
            best_centers, time_budget=min(15.0, deadline - time.time() - 50)
        )
        
    # 4. Smooth Max-Min Polish (Structural Change)
    # This replaces the binary search + violation polish
    if time.time() < deadline - 20:
        best_centers, best_radius = _smooth_polish(
            best_centers, time_budget=min(50.0, deadline - time.time() - 5)
        )
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
