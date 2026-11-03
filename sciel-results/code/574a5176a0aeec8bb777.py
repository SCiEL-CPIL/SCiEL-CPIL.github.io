"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time

N = 50
DEADLINE = 118.0  # safety margin before 120s kill (evaluator kills at 150s)

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
    c: (N, 2) array of centers
    R: target radius
    iters: number of relaxation iterations
    eta0, eta1: learning rate start and end
    t0, deadline: optional time tracking
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
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    Returns (best_centers, best_radius).
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

def _soft_min_radius(pos, alpha=50.0):
    """Smooth surrogate for the true radius using log-sum-exp soft-min."""
    pos = np.asarray(pos, dtype=np.float64)
    # Wall distances
    wall = 0.5 - np.abs(pos)
    wall_vals = np.minimum(wall[:, 0], wall[:, 1])
    # Pairwise half-distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    # Combine all constraints into a flat array
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals.ravel()])
    # Soft-min via log-sum-exp
    m = all_vals.min()
    s = np.log(np.sum(np.exp(alpha * (all_vals - m)))) / alpha
    return m + s

def _soft_min_radius_grad(pos, alpha=50.0):
    """Gradient of the soft-min radius surrogate."""
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    # Wall distances
    wall = 0.5 - np.abs(pos)
    wall_vals = np.minimum(wall[:, 0], wall[:, 1])
    # Pairwise half-distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, 1e-12)
    pair_vals = dist / 2.0
    # Combine
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals.ravel()])
    m = all_vals.min()
    shifted = alpha * (all_vals - m)
    exps = np.exp(shifted)
    weights = exps / exps.sum()
    
    # Gradient wrt pos
    grad = np.zeros_like(pos)
    # Wall contribution
    w_wall = weights[:n]
    for i in range(n):
        for k in range(2):
            if pos[i, k] > 0:
                if wall[i, k] <= wall[i, 1-k]:
                    grad[i, k] -= w_wall[i]
            else:
                if wall[i, k] <= wall[i, 1-k]:
                    grad[i, k] += w_wall[i]
    # Pair contribution
    w_pair = weights[n:].reshape(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dist[i, j] > 1e-12:
                grad[i] -= w_pair[i, j] * diff[i, j] / dist[i, j]
    return grad

def _scipy_polish(c, r, t0, deadline, maxiter=200, alpha=50.0):
    """Polish using L-BFGS-B on the soft-min surrogate."""
    from scipy.optimize import minimize
    
    def obj(x):
        return -_soft_min_radius(x, alpha)
    
    def obj_grad(x):
        return -_soft_min_radius_grad(x, alpha)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    x0 = c.flatten()
    
    try:
        res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        c_new = res.x.reshape(N, 2)
        r_new = _score(c_new)
        if r_new > r and np.all(np.isfinite(c_new)):
            return c_new, r_new
    except Exception:
        pass
    return c, r

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    c: centers, r: current radius
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        # Langevin update: deterministic + stochastic
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
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
    # Allocate time: ~15s per init for growth, then refine
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 20) / n_inits  # reserve 20s for final refinement
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1500, 0.02, 0.002, t0, DEADLINE)
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        # Check time
        if time.time() - t0 > DEADLINE - 25:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Scipy L-BFGS-B polish on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c, r = _scipy_polish(best, best_r, t0, DEADLINE, maxiter=300, alpha=50.0)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 2b: Langevin refinement after scipy polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=5000, T0=0.0015, T1=0.00005, lr=0.003)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 3: Final fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        r_cur = _score(best)
        c = _relax(best, r_cur, 3000, 0.01, 0.0005, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Phase 4: Final scipy polish if time allows
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _scipy_polish(best, best_r, t0, DEADLINE, maxiter=200, alpha=80.0)
        if r > best_r:
            best_r = r
            best = c
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
