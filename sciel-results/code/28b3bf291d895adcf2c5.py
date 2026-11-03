"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    # Pairwise distances (upper triangle only for efficiency)
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    # Wall distances
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)

def _relax(c, R, iters, eta0, eta1):
    """Relaxation: push circles apart at target radius R, handle walls."""
    c = c.copy()
    n = len(c)
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        # Pairwise forces
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        # Force magnitude: push apart if closer than 2R
        mag = np.maximum(2.0 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, None] * diff).sum(axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """Grow radius from R0 to R1, keeping the best configuration found."""
    c = start.copy()
    best_c = c.copy()
    best_r = _radius(c)
    R = R0
    while R <= R1:
        if (time.time() - t0) > deadline - 5:
            break
        c = _relax(c, R, iters, eta0, eta1)
        r = _radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best_c, best_r

def _slsqp_polish(c_init, r_init, t0, deadline):
    """
    Polish the configuration using SLSQP to maximize radius r.
    Variables: x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, r]
    Objective: minimize -r
    Constraints:
      1. c_ix + r <= 0.5
      2. -c_ix + r <= 0.5  => c_ix >= r - 0.5
      3. c_iy + r <= 0.5
      4. -c_iy + r <= 0.5
      5. ||c_i - c_j||^2 - (2r)^2 >= 0 for all i < j
    """
    n = len(c_init)
    # Initial guess
    x0 = np.concatenate([c_init.flatten(), [r_init]])
    
    # Objective function
    def objective(x):
        r = x[-1]
        return -r

    # Constraints (inequality: fun(x) >= 0)
    cons = []
    
    # Wall constraints
    # c_x + r <= 0.5  => 0.5 - c_x - r >= 0
    # -c_x + r <= 0.5 => -0.5 - r + c_x >= 0  (Wait, c_x - r >= -0.5 is c_x + 0.5 - r >= 0? No.)
    # Square is [-0.5, 0.5].
    # c_x <= 0.5 - r  => 0.5 - r - c_x >= 0
    # c_x >= -0.5 + r => c_x - r + 0.5 >= 0
    # c_y <= 0.5 - r  => 0.5 - r - c_y >= 0
    # c_y >= -0.5 + r => c_y - r + 0.5 >= 0
    
    # To speed up SLSQP, we can define constraints in a vectorized way if possible, 
    # but scipy SLSQP usually expects a list of dicts. 
    # Given N=50, there are 1225 pair constraints + 200 wall constraints. 
    # This is a lot for SLSQP. We might need to be careful with the time limit.
    
    # Let's implement the constraints efficiently.
    
    # Wall constraints
    def wall_cons(x):
        c = x[:2*n].reshape(n, 2)
        r = x[-1]
        # 0.5 - r - c_x >= 0
        # c_x - r + 0.5 >= 0
        # 0.5 - r - c_y >= 0
        # c_y - r + 0.5 >= 0
        return np.concatenate([
            0.5 - r - c[:, 0],
            c[:, 0] - r + 0.5,
            0.5 - r - c[:, 1],
            c[:, 1] - r + 0.5
        ])
        
    cons.append({'type': 'ineq', 'fun': wall_cons})
    
    # Pairwise constraints
    iu = np.triu_indices(n, k=1)
    idx_i = iu[0]
    idx_j = iu[1]
    
    def pair_cons(x):
        c = x[:2*n].reshape(n, 2)
        r = x[-1]
        ci = c[idx_i]
        cj = c[idx_j]
        diff = ci - cj
        dist_sq = (diff ** 2).sum(axis=1)
        # dist_sq - (2r)^2 >= 0
        return dist_sq - 4.0 * r * r
        
    cons.append({'type': 'ineq', 'fun': pair_cons})
    
    # Bounds for centers and radius
    # Centers must be within [-0.5, 0.5]
    # Radius must be > 0
    bounds = [(-0.5, 0.5)] * (2 * n)
    bounds.append((0.0, 0.5))
    
    # Options
    # maxiter is important. SLSQP can be slow.
    # We will run it in a loop or with a time check if possible, but scipy minimize doesn't support time limits directly.
    # We will set a reasonable maxiter and rely on the external deadline check to interrupt if needed?
    # Actually, we can't easily interrupt minimize. We should estimate time.
    # For N=50, SLSQP with ~1400 constraints might take a few seconds per iteration. 
    # Let's limit maxiter to 50-100.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if res.success:
            c_opt = res.x[:2*n].reshape(n, 2)
            r_opt = res.x[-1]
            # Validate
            if np.all(np.isfinite(c_opt)) and r_opt > 0:
                # Check if it's actually better
                r_calc = _radius(c_opt)
                if r_calc >= r_init - 1e-10:
                    return c_opt, r_calc
        else:
            # If SLSQP fails, try a smaller step or just return original
            pass
            
    except Exception:
        pass
        
    return c_init, r_init

def _init_7x7_plus():
    """7x7 grid (49 points) plus one extra at origin."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    """8x8 grid with jitter, pick 50 points."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c

def _init_rand(rng):
    """Random uniform initialization."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def construct():
    t0 = time.time()
    # 150s limit. Leave 5s for safety.
    deadline = t0 + 145.0
    
    rng = np.random.default_rng(12345)

    # Multiple diverse initializations
    inits = [
        _init_7x7_plus(),
        _init_jitter8(rng),
        _init_rand(rng),
    ]

    best_c = None
    best_r = -1.0

    # Phase 1: Growth/Relaxation to find a good basin
    # We allocate about 80% of the time to this phase to explore different basins.
    # The relaxation is fast, so we can do many steps.
    
    for start in inits:
        if (time.time() - t0) > deadline - 30:  # Reserve 30s for polish
            break
        # Growth phase: expand from small radius to near-optimal
        # We use a slightly smaller R1 to ensure we don't spend too much time here,
        # as the polish will handle the final push.
        c, r = _grow(start, 0.04, 0.0715, 0.0003, 1200, 0.025, 0.002, t0, deadline - 30)
        if r > best_r:
            best_r = r
            best_c = c

    if best_c is None:
        # Fallback: return 7x7 grid
        best_c = _init_7x7_plus()
        best_r = _radius(best_c)

    # Phase 2: SLSQP Polish
    # We have ~30s left. SLSQP can be slow.
    # We will try to polish the best candidate found.
    # If SLSQP takes too long, we might not finish, but we have a deadline guard in the outer loop?
    # No, minimize blocks. We must be careful.
    # For N=50, SLSQP with 1400 constraints is heavy.
    # Let's try to run it. If it's too slow, we might need to reduce constraints or use a different method.
    # However, the inspiration suggests SLSQP works well.
    
    # We will attempt SLSQP polish.
    # To be safe against timeout, we can't easily interrupt minimize.
    # But we can check if we have enough time.
    if (time.time() - t0) < deadline - 5:
        c_polished, r_polished = _slsqp_polish(best_c, best_r, t0, deadline)
        if r_polished > best_r:
            best_c = c_polished
            best_r = r_polished
            
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-15, 0.5 - 1e-15)
    if not np.all(np.isfinite(best_c)) or _radius(best_c) <= 0:
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_c
# EVOLVE-BLOCK-END
