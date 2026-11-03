"""Pack 50 equal circles in a unit square, maximize common radius.

Multi-stage approach:
1. Multiple initializations (7x7 grid, jittered 8x8, hexagonal, random)
2. Hard-disk radius expansion to reach near-optimal density
3. SLSQP epigraph polish with active-set reduction
4. Final high-precision deterministic local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, beta=500.0):
    """
    Smooth surrogate for min radius using log-sum-exp.
    Returns a differentiable approximation to the true min radius.
    Higher beta = sharper (closer to true min).
    """
    c = np.asarray(c)
    # Wall distances
    wall = 0.5 - np.abs(c)  # (N, 2)
    # Pair distances / 2
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0  # (N, N)
    
    # Combine all constraints into one vector
    all_vals = np.concatenate([wall.ravel(), pair[~np.eye(N, dtype=bool).ravel()]/2.0])
    
    # Log-sum-exp soft min: -logsumexp(-beta * x) / beta
    # For numerical stability, shift by max
    m = np.max(all_vals)
    shifted = -beta * (all_vals - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    lse = np.log(np.sum(np.exp(shifted)))
    soft_min = m - lse / beta
    return soft_min


def _soft_min_radius_neg(c, beta=500.0):
    """Negative soft-min radius (for minimization)."""
    return -_soft_min_radius(c, beta)


def _soft_min_radius_grad(c, beta=500.0):
    """
    Gradient of the soft-min radius with respect to centres.
    Uses the softmax weights from the log-sum-exp.
    """
    c = np.asarray(c)
    wall = 0.5 - np.abs(c)  # (N, 2)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0  # (N, N)
    
    all_vals = np.concatenate([wall.ravel(), pair[~np.eye(N, dtype=bool).ravel()]/2.0])
    
    m = np.max(all_vals)
    shifted = -beta * (all_vals - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= weights.sum()
    
    # Gradient of -logsumexp(-beta * x) / beta with respect to x
    # = -sum(w_i * (-beta) * (dx_i/dx)) / beta = sum(w_i * dx_i/dx)
    # where dx_i/dx is the gradient of each constraint w.r.t. c
    
    grad = np.zeros_like(c)
    
    # Wall contributions
    n_wall = wall.size
    w_wall = weights[:n_wall].reshape(N, 2)
    # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad -= w_wall * signs
    
    # Pair contributions
    w_pair = weights[n_wall:].reshape(N, N)
    # d||c_i - c_j||/2 / dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    diff = c[:, None, :] - c[None, :, :]
    norms = np.sqrt(np.sum(diff ** 2, axis=-1))
    norms[~np.eye(N, dtype=bool)] = np.maximum(norms[~np.eye(N, dtype=bool)], 1e-12)
    # grad_i += sum_j w_ij * (c_i - c_j) / (2 * ||c_i - c_j||)
    unit = diff / (norms[:, :, None] * 2.0)
    np.fill_diagonal(unit, 0.0)
    grad += np.sum(w_pair[:, :, None] * unit, axis=1)
    
    return grad


def _kissing_graph(c, r, tol=1e-3):
    """
    Find active constraints: pairs at distance ~2r and circles touching walls.
    Returns list of (i, j) for pairs and (i, k) for walls.
    """
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    wall = 0.5 - np.abs(c)
    
    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(d[i, j] - 2 * r) < tol:
                active_pairs.append((i, j))
    
    active_walls = []
    for i in range(N):
        for k in range(2):
            if wall[i, k] < r + tol:
                active_walls.append((i, k))
    
    return active_pairs, active_walls


def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=30.0):
    """
    SLSQP polish using epigraph formulation.
    Variables: [x_1, y_1, ..., x_N, y_N, r]
    Maximize r subject to:
      - ||c_i - c_j|| >= 2r for all i < j
      - 0.5 - |c_ik| >= r for all i, k
    """
    
    def objective(x):
        return -x[-1]  # minimize -r
    
    def objective_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g
    
    def _pair_constraint(x, i, j):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        return np.linalg.norm(c[i] - c[j]) - 2 * r
    
    def _wall_constraint(x, i, k):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        return 0.5 - np.abs(c[i, k]) - r
    
    def _pair_jac(x, i, j):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        g = np.zeros(2 * N + 1)
        dij = c[i] - c[j]
        norm = np.linalg.norm(dij)
        if norm > 1e-12:
            g[i*2] = dij[0] / norm
            g[i*2+1] = dij[1] / norm
            g[j*2] = -dij[0] / norm
            g[j*2+1] = -dij[1] / norm
        g[-1] = -2.0
        return g
    
    def _wall_jac(x, i, k):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        g = np.zeros(2 * N + 1)
        sign = np.sign(c[i, k])
        if sign == 0:
            sign = 1.0
        g[i*2 + k] = -sign
        g[-1] = -1.0
        return g
    
    # Build constraint list
    constraint_list = []
    for i in range(N):
        for j in range(i + 1, N):
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: _pair_constraint(x, i, j),
                'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)
            })
    for i in range(N):
        for k in range(2):
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, i=i, k=k: _wall_constraint(x, i, k),
                'jac': lambda x, i=i, k=k: _wall_jac(x, i, k)
            })
    
    x0 = np.concatenate([c_init.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    
    # Run SLSQP with multiple maxiter settings
    for maxiter in [500, 1000, 2000]:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget:
            break
        
        try:
            res = minimize(
                objective,
                best_x,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            if res.success or res.fun < objective(best_x):
                r_new = res.x[-1]
                c_new = res.x[:2*N].reshape(N, 2)
                r_check = _radius_fast(c_new)
                if r_check > best_r:
                    best_r = r_check
                    best_x = res.x.copy()
        except Exception:
            break
    
    return best_x[:2*N].reshape(N, 2), best_r


def _lbfgs_polish(c_init, r_init, t0, time_budget=25.0):
    """
    L-BFGS-B polish using soft-min surrogate.
    This handles the nonsmooth min by smoothing it, allowing
    gradient-based optimization to make progress.
    """
    
    def objective(x):
        c = x.reshape(N, 2)
        return -_soft_min_radius(c, beta=500.0)
    
    def objective_grad(x):
        c = x.reshape(N, 2)
        g = -_soft_min_radius_grad(c, beta=500.0)
        return g.ravel()
    
    x0 = c_init.ravel()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_c = c_init.copy()
    best_r = r_init
    
    # Try with different beta values (sharper = closer to true min)
    for beta in [300.0, 500.0, 800.0]:
        if time.time() - t0 > DEADLINE - 15:
            break
        
        def obj_beta(x):
            c = x.reshape(N, 2)
            return -_soft_min_radius(c, beta=beta)
        
        def grad_beta(x):
            c = x.reshape(N, 2)
            return -_soft_min_radius_grad(c, beta=beta).ravel()
        
        try:
            res = minimize(
                obj_beta,
                x0,
                jac=grad_beta,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                x0 = res.x.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square.  Vectorized.
    """
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
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _active_set_refine(c, r, max_iter=100, tol=1e-10):
    """
    Active-set refinement: identify the kissing graph (pairs at distance ~2r
    and circles touching walls), then solve a local QP-like problem to
    maximize the minimum constraint while keeping active constraints tight.
    Uses a gradient-based approach on the active set.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    for _ in range(max_iter):
        # Find active constraints
        active_pairs, active_walls = _kissing_graph(c, r_cur, tol=1e-4)
        
        if not active_pairs and not active_walls:
            break
        
        # Build gradient ascent direction to increase min radius
        # For each active constraint, push circles apart or away from wall
        grad = np.zeros((N, 2))
        
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm > 1e-12:
                g = dij / norm
                grad[i] += 0.5 * g
                grad[j] -= 0.5 * g
        
        for (i, k) in active_walls:
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            grad[i, k] += -0.5 * sign
        
        # Normalize gradient
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-12:
            break
        grad /= norm_grad
        
        # Line search with more steps for finer resolution
        best_step = 0
        best_r = r_cur
        for step in np.linspace(0, 0.003, 40):
            c_try = c + step * grad
            c_try = np.clip(c_try, -0.5 + 1e-9, 0.5 - 1e-9)
            r_try = _radius_fast(c_try)
            if r_try > best_r:
                best_r = r_try
                best_step = step
        
        if best_step == 0:
            break
        
        c = c + best_step * grad
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        r_cur = _radius_fast(c)
        
        if best_step < tol:
            break
    
    return c, _radius_fast(c)


def _final_local_search(c, r, max_iter=500, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in multiple directions, keeping the best improvement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Generate directions: 24 angular directions
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in dirs:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
                    c = c_try
                    r_cur = r_try
                    improved = True
        if not improved:
            break
    return c, r_cur


def _init_7x7_plus():
    """7x7 grid (49 points) + 50th near centre."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jittered_8x8(rng):
    """8x8 grid, take 50, jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.015, 0.015, c.shape)
    return c


def _init_hexagonal(rng):
    """Hexagonal close-packing initialization."""
    s = 0.12
    pts = []
    for row in range(8):
        y = -0.45 + row * s * np.sqrt(3) / 2
        offset = (s / 2) if (row % 2 == 1) else 0.0
        for col in range(8):
            x = -0.45 + col * s + offset
            if abs(x) <= 0.45 and abs(y) <= 0.45:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) > N:
        idx = rng.permutation(len(pts))[:N]
        pts = pts[idx]
    elif len(pts) < N:
        extra = rng.uniform(-0.4, 0.4, (N - len(pts), 2))
        pts = np.vstack([pts, extra])
    pts += rng.uniform(-0.01, 0.01, pts.shape)
    return np.clip(pts, -0.45, 0.45)


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, (N, 2))


def _grow_radius(start, R0, R1, step, iters_per_step, eta0, eta1, t0):
    """
    Gradually increase the target radius R from R0 to R1,
    running hard-disk relaxation at each step.
    """
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 20:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Multiple initializations
    inits = [
        _init_7x7_plus(),
        _init_jittered_8x8(rng),
        _init_hexagonal(rng),
        _init_random(rng),
    ]

    best_c = None
    best_r = -1.0

    for start in inits:
        if (time.time() - t0) > DEADLINE - 50:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 500, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2a: L-BFGS-B polish with soft-min surrogate
    if (time.time() - t0) < DEADLINE - 35:
        c_lbfgs, r_lbfgs = _lbfgs_polish(
            best_c, best_r, t0,
            time_budget=min(25.0, DEADLINE - 35 - (time.time() - t0))
        )
        if r_lbfgs > best_r:
            best_r = r_lbfgs
            best_c = c_lbfgs

    # Phase 2b: SLSQP epigraph polish
    if (time.time() - t0) < DEADLINE - 25:
        c_polished, r_polished = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(30.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: Active-set refinement
    if (time.time() - t0) < DEADLINE - 15:
        c_as, r_as = _active_set_refine(best_c, best_r, max_iter=200)
        if r_as > best_r:
            best_r = r_as
            best_c = c_as

    # Phase 4: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 8:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=400, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        # One more round with even smaller steps
        if (time.time() - t0) < DEADLINE - 2:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=300, step_scale=3e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
