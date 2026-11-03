"""
Circle packing in a unit square: maximize common radius for 50 equal circles.

Strategy:
1. Multiple diverse initializations.
2. Grow-and-relax to establish global structure.
3. Hill-climbing local search to escape local minima.
4. Vectorized L-BFGS-B polish with a soft-min objective for high precision.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 150s kill (evaluator limit)

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        # Force magnitude: push apart if d < 2R
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls: push inward if outside [-(0.5-R), 0.5-R]
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best

def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c

def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _local_search(centers, max_iter=2000, time_limit=20.0, step_init=0.01, t0=None, deadline=None):
    """
    Hill-climbing local search: move each circle away from its nearest constraint.
    """
    centers = centers.copy()
    best_r = _score(centers)
    step_size = step_init
    no_improve = 0
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 5:
                break
        if time.time() - (t0 if t0 else time.time()) > time_limit:
             break
            
        improved = False
        indices = rng.permutation(N)
        for idx in indices:
            orig = centers[idx].copy()
            # Current local min for this circle
            wall_d = 0.5 - np.abs(orig)
            min_wall = np.min(wall_d)
            diffs = centers - orig
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[idx] = np.inf
            min_pair = np.min(dists) / 2.0
            cur_local = min(min_wall, min_pair)
            
            # Determine push direction: away from closest constraint
            closest_idx = np.argmin(dists)
            if dists[closest_idx] / 2.0 < min_wall:
                direction = orig - centers[closest_idx]
            else:
                # Move toward center (away from nearest wall)
                direction = -orig
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = rng.normal(0, 1, 2)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Try several perturbations
            best_local = cur_local
            best_new = orig.copy()
            for _ in range(10): # Reduced trials for speed
                noise = rng.normal(0, 1, 2)
                move = direction * step_size + noise * step_size * 0.4
                new_c = np.clip(orig + move, -0.4999, 0.4999)
                # Quick local min
                nw = 0.5 - np.abs(new_c)
                nmin_wall = np.min(nw)
                nd = centers - new_c
                ndists = np.sqrt(np.sum(nd ** 2, axis=1))
                ndists[idx] = np.inf
                nmin_pair = np.min(ndists) / 2.0
                nlocal = min(nmin_wall, nmin_pair)
                if nlocal > best_local + 1e-12:
                    best_local = nlocal
                    best_new = new_c
            if best_local > cur_local + 1e-12:
                centers[idx] = best_new
                improved = True
        
        if not improved:
            no_improve += 1
            step_size *= 0.95
            if no_improve > 30:
                break
        else:
            no_improve = 0
            
        r = _score(centers)
        if r > best_r:
            best_r = r
            
    return centers, best_r

def _softmin_obj_and_grad(x, beta=100.0):
    """
    Vectorized soft-min objective and gradient for L-BFGS-B.
    Objective: minimize -soft_min(distances) / 2.
    x: shape (N*2,)
    """
    c = x.reshape(N, 2)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(N, dtype=bool)
    dists = np.sqrt(dists_sq[mask])
    
    # Wall distances
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    wall_dists = np.minimum(wx, wy)
    
    all_dists = np.concatenate([dists, wall_dists])
    
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    weights = exp_vals / sum_exp
    
    f = -soft_min / 2.0
    
    grad_c = np.zeros((N, 2))
    
    # Pair gradients
    num_pairs = N * (N - 1) // 2
    pair_weights = weights[:num_pairs]
    i_idx, j_idx = np.where(mask)
    
    pair_diff = c[i_idx] - c[j_idx]
    pair_dists = dists
    safe_dists = np.maximum(pair_dists, 1e-12)
    pair_dirs = pair_diff / safe_dists[:, np.newaxis]
    
    # Vectorized accumulation
    # grad_c[i] += w * dir, grad_c[j] -= w * dir
    # Using bincount or scatter-add logic is tricky with 2D, but we can do:
    # Actually, for N=50, the loop is ~1225 iterations. It's fast enough in Python if not in a tight inner loop of optimization?
    # L-BFGS calls this many times. Let's try to vectorize.
    
    # Alternative: 
    # grad_c = np.zeros((N,2))
    # np.add.at(grad_c, i_idx, pair_weights[:, None] * pair_dirs)
    # np.add.at(grad_c, j_idx, -pair_weights[:, None] * pair_dirs)
    
    np.add.at(grad_c, i_idx, pair_weights[:, np.newaxis] * pair_dirs)
    np.add.at(grad_c, j_idx, -pair_weights[:, np.newaxis] * pair_dirs)
    
    # Wall gradients
    wall_weights = weights[num_pairs:]
    # wx <= wy means x is the limiting factor
    x_lim = wx <= wy
    # If x_lim, grad is +/- 1 in x direction depending on sign of c[i,0]
    # If not x_lim, grad is +/- 1 in y direction depending on sign of c[i,1]
    
    # x component
    sign_x = np.sign(c[:, 0])
    # If x_lim is True, add weight * sign_x to grad_c[:, 0]
    grad_c[:, 0] += wall_weights * sign_x * x_lim
    # If x_lim is False, add 0 to grad_c[:, 0]
    
    # y component
    sign_y = np.sign(c[:, 1])
    # If x_lim is False, add weight * sign_y to grad_c[:, 1]
    grad_c[:, 1] += wall_weights * sign_y * (~x_lim)
    
    g = (0.5 * grad_c).flatten()
    
    return f, g

def _scipy_polish(centers, t0, deadline, maxiter=300):
    """
    Polish using scipy L-BFGS-B with smooth soft-min objective.
    """
    if time.time() - t0 > deadline - 10.0:
        return centers
    
    x0 = centers.flatten().copy()
    x0 = np.clip(x0, -0.5, 0.5)
    
    bounds = [(-0.5, 0.5)] * (N * 2)
    best_c = centers.copy()
    best_r = _score(centers)
    
    # Try different betas
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - t0 > deadline - 5.0:
            break
        
        try:
            result = minimize(
                _softmin_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            
            x_opt = result.x
            if np.all(np.isfinite(x_opt)):
                c_opt = x_opt.reshape(N, 2)
                c_opt = np.clip(c_opt, -0.5, 0.5)
                r_opt = _score(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt.copy()
                    x0 = x_opt.copy() # Use best found as start for next beta
        except Exception:
            pass
    
    return best_c

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    # Reserve 40s for local search and polish
    grow_deadline = t0 + (DEADLINE - 40)
    
    for i, start in enumerate(inits):
        remaining = grow_deadline - (time.time() - t0)
        if remaining < 5:
            break
        
        # Dynamic step size for grow
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1000, 0.02, 0.002, t0, grow_deadline)
        
        if r > best_r:
            best_r = r
            best = c.copy()
            
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Local Search (Hill Climbing)
    # Use the best from grow
    local_search_deadline = t0 + (DEADLINE - 15) # Reserve 15s for polish
    c_ls, r_ls = _local_search(best, max_iter=2000, time_limit=15.0, step_init=0.01, t0=t0, deadline=local_search_deadline)
    if r_ls > best_r:
        best_r = r_ls
        best = c_ls
    
    # Phase 3: Gradient Polish
    # Use remaining time
    c_pol, r_pol = _scipy_polish(best, t0, DEADLINE)
    if r_pol > best_r:
        best_r = r_pol
        best = c_pol
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best

if __name__ == "__main__":
    # Quick self-test
    t_start = time.time()
    res = construct()
    t_end = time.time()
    r = _score(res)
    print(f"Time: {t_end - t_start:.2f}s, Radius: {r:.10f}")
