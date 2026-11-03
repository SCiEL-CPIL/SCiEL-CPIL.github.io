"""Pack 50 equal circles in a unit square, maximize common radius.

Uses Iterated Local Search (ILS) to escape local minima, followed by
high-precision polishing with L-BFGS-B (soft-min) and SLSQP (epigraph).
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 118.0  # safety margin before 150s kill

def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)

def _soft_min_radius(c, t):
    """
    Differentiable surrogate: soft-min of all pairwise half-distances
    and wall distances.
    """
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()
    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return soft

def _soft_min_radius_neg(x, t):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)

def _soft_min_radius_grad(x, t):
    """Gradient of negative soft-min radius w.r.t. x (flattened centres)."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()
    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()
    grad_flat = np.zeros(2 * N)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad
    dist = d.copy()
    np.fill_diagonal(dist, 1.0)
    for i in range(N):
        for j in range(i + 1, N):
            idx_ij = i * N + j
            idx_ji = j * N + i
            w_ij = exps[2 * N + idx_ij] / Z
            w_ji = exps[2 * N + idx_ji] / Z
            total_w = w_ij + w_ji
            if total_w > 1e-15:
                dij = c[i] - c[j]
                norm = dist[i, j]
                if norm > 1e-12:
                    g = dij / (2.0 * norm)
                    grad_flat[i * 2] += total_w * g[0]
                    grad_flat[i * 2 + 1] += total_w * g[1]
                    grad_flat[j * 2] -= total_w * g[0]
                    grad_flat[j * 2 + 1] -= total_w * g[1]
    return grad_flat

def _relax_hard_disk(c, R, iters, eta0, eta1):
    """Hard-disk relaxation."""
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _quick_lbfgs_polish(c, t0, time_budget=2.0):
    """Quick L-BFGS-B polish for ILS loop."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    # Use a moderate temperature for quick convergence
    t = 5.0 
    if (time.time() - t0) > time_budget:
        return c, best_r
    try:
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
    except Exception:
        pass
    return best_x.reshape(N, 2), best_r

def _lbfgs_polish(c, t0, time_budget=15.0):
    """Full L-BFGS-B polish with multi-temperature."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)
    temps = [50.0, 20.0, 10.0, 5.0, 2.0]
    for t in temps:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                lambda xx: _soft_min_radius_neg(xx, t),
                x,
                jac=lambda xx: _soft_min_radius_grad(xx, t),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_x = res.x.copy()
                x = res.x.copy()
        except Exception:
            pass
    return best_x.reshape(N, 2), best_r

def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=30.0):
    """SLSQP polish using epigraph formulation."""
    x0 = np.concatenate([c_init.ravel(), [r_init]])
    pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
    wall_indices = [(i, k) for i in range(N) for k in range(2)]
    n_pairs = len(pair_indices)
    n_walls = len(wall_indices)
    n_cons = n_pairs + n_walls

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints_fun(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        cons = np.zeros(n_cons)
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        row = 0
        for (i, j) in pair_indices:
            cons[row] = d[i, j] - 2 * r
            row += 1
        wall_dist = 0.5 - np.abs(c)
        for (i, k) in wall_indices:
            cons[row] = wall_dist[i, k] - r
            row += 1
        return cons

    def constraints_jac(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        J = np.zeros((n_cons, 2 * N + 1))
        d = cdist(c, c)
        np.fill_diagonal(d, 1.0)
        row = 0
        for (i, j) in pair_indices:
            dij = c[i] - c[j]
            norm = d[i, j]
            if norm > 1e-12:
                J[row, 2 * i] = dij[0] / norm
                J[row, 2 * i + 1] = dij[1] / norm
                J[row, 2 * j] = -dij[0] / norm
                J[row, 2 * j + 1] = -dij[1] / norm
            J[row, -1] = -2.0
            row += 1
        for (i, k) in wall_indices:
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            J[row, 2 * i + k] = -sign
            J[row, -1] = -1.0
            row += 1
        return J

    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
    cons = [{'type': 'ineq', 'fun': constraints_fun, 'jac': constraints_jac}]

    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-14, 'maxls': 50}
        )
        c_new = res.x[:2 * N].reshape(N, 2)
        r_check = _radius_fast(c_new)
        if r_check > r_init:
            return c_new, r_check
    except Exception:
        pass
    return c_init, r_init

def _init_7x7_plus():
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.02, 0.02]])
    return np.vstack([g, extra])

def _init_jittered_8x8(rng):
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.02, 0.02, c.shape)
    return c

def _init_hexagonal(rng):
    c = np.zeros((N, 2))
    dx = 1.0 / 7
    dy = dx * np.sqrt(3) / 2
    row = 0
    col = 0
    count = 0
    while count < N:
        x = -0.5 + dx / 2 + col * dx
        y = -0.5 + dy / 2 + row * dy
        if row % 2 == 1:
            x += dx / 2
        if -0.5 < x < 0.5 and -0.5 < y < 0.5:
            c[count] = [x, y]
            count += 1
        col += 1
        if col > 8:
            col = 0
            row += 1
    c += rng.uniform(-0.01, 0.01, c.shape)
    return c

def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
    ]
    
    best_c = None
    best_r = -1.0
    
    # Phase 1: Initial relaxation to get decent local minima
    for start in inits:
        if (time.time() - t0) > DEADLINE - 40:
            break
        # Grow radius to a reasonable level
        c = _relax_hard_disk(start, 0.06, 1000, 0.01, 0.001)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: Iterated Local Search (ILS)
    # We perturb the best solution, relax, and polish quickly.
    # This helps escape local minima found in Phase 1.
    noise_scale = 0.01
    stall_count = 0
    
    while (time.time() - t0) < DEADLINE - 30:
        # Perturb
        c_pert = best_c + rng.normal(0, noise_scale, best_c.shape)
        c_pert = np.clip(c_pert, -0.49, 0.49)
        
        # Relax to fix overlaps
        # Use current best radius as target for relaxation
        c_relaxed = _relax_hard_disk(c_pert, best_r, 200, 0.01, 0.001)
        
        # Quick polish
        c_polished, r_polished = _quick_lbfgs_polish(c_relaxed, t0, time_budget=1.5)
        
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished
            stall_count = 0
            # Decrease noise slightly when improving
            noise_scale *= 0.95
        else:
            stall_count += 1
            # Increase noise if stuck
            if stall_count > 5:
                noise_scale = min(noise_scale * 1.1, 0.05)
                stall_count = 0

    # Phase 3: Full Polish
    if (time.time() - t0) < DEADLINE - 20:
        c_polished, r_polished = _lbfgs_polish(best_c, t0, time_budget=min(15.0, DEADLINE - 20 - (time.time() - t0)))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 4: SLSQP Epigraph Polish
    if (time.time() - t0) < DEADLINE - 5:
        c_slq, r_slq = _slsqp_epigraph_polish(best_c, best_r, t0, time_budget=min(30.0, DEADLINE - 5 - (time.time() - t0)))
        if r_slq > best_r:
            best_r = r_slq
            best_c = c_slq

    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_c
