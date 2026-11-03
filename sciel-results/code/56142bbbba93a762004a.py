"""Pack 50 equal circles in a unit square, maximize common radius.

Strategy:
1. Multiple diverse initializations
2. Hard-disk relaxation with adaptive radius growth
3. SLSQP epigraph polish (primary high-precision method)
4. L-BFGS-B soft-min polish
5. Contact-graph-based active set refinement
6. Final high-precision local search
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """Differentiable soft-min of all constraints."""
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
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()
    
    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-(all_terms - m) / t)
    Z = exps.sum()
    
    grad_flat = np.zeros(2 * N)
    
    # Wall gradient
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += (exps[:2 * N] / Z) * w_grad
    
    # Pairwise gradient
    weights_pair = exps[2 * N:].reshape(N, N)
    
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        
        d_i = d[i, mask]
        diff_i = diff[i, mask]
        w_i = weights_pair[i, mask]
        
        valid = d_i > 1e-12
        if np.any(valid):
            factor = np.zeros_like(d_i)
            factor[valid] = w_i[valid] / (2.0 * d_i[valid])
            grad_i = np.sum(factor[:, None] * diff_i, axis=0)
            grad_flat[i * 2] += grad_i[0]
            grad_flat[i * 2 + 1] += grad_i[1]
            
    return grad_flat


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """Hard-disk relaxation with vectorized forces."""
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


def _voronoi_style_refine(c, r, t0, time_budget=25.0):
    """
    Voronoi-style refinement: for each circle, find its nearest neighbors
    and optimize its position to maximize the minimum distance to those
    neighbors and the walls. This is a per-circle local optimization.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    best_c = c.copy()
    best_r = r_cur

    for it in range(3):
        if (time.time() - t0) > t0 + time_budget:
            break
        for i in range(N):
            if (time.time() - t0) > t0 + time_budget:
                break
            # Find nearest neighbors for circle i
            d = cdist(c[i:i+1], c).ravel()
            d[i] = np.inf
            # Get 8 nearest neighbors
            nn_idx = np.argsort(d)[:8]
            
            # Optimize position of circle i
            def obj_i(x):
                c_try = c.copy()
                c_try[i] = x
                # Distance to neighbors
                min_nn = np.min(cdist(c_try[i:i+1], c_try[nn_idx]).ravel())
                # Distance to walls
                min_wall = np.min(0.5 - np.abs(x))
                return -min(min_nn / 2.0, min_wall)
            
            def grad_i(x):
                c_try = c.copy()
                c_try[i] = x
                min_nn = np.min(cdist(c_try[i:i+1], c_try[nn_idx]).ravel())
                min_wall = np.min(0.5 - np.abs(x))
                r = min(min_nn / 2.0, min_wall)
                
                g = np.zeros(2)
                # Gradient from nearest neighbor
                diffs = c_try[nn_idx] - x
                dists = np.linalg.norm(diffs, axis=1)
                min_dist = np.min(dists)
                min_idx = np.argmin(dists)
                if min_dist > 1e-12 and min_nn / 2.0 <= min_wall:
                    g = -diffs[min_idx] / (2.0 * min_dist)
                else:
                    # Gradient from wall
                    sign = np.sign(x)
                    sign[sign == 0] = 1.0
                    g = -sign
                return g
            
            x0 = c[i].copy()
            bounds = [(-0.5, 0.5), (-0.5, 0.5)]
            
            try:
                res = minimize(
                    obj_i,
                    x0,
                    jac=grad_i,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.success:
                    c_new = c.copy()
                    c_new[i] = res.x
                    r_new = _radius_fast(c_new)
                    if r_new > best_r + 1e-14:
                        best_r = r_new
                        best_c = c_new
                        c = c_new
            except Exception:
                pass
    
    return best_c, best_r


def _slsqp_epigraph_polish(c_init, r_init, t0, time_budget=35.0):
    """
    SLSQP polish using epigraph formulation.
    This is the primary high-precision method.
    """
    x0 = np.concatenate([c_init.ravel(), [r_init]])

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        cons = []
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                cons.append(d[i, j] - 2 * r)
        wall_dist = 0.5 - np.abs(c)
        for i in range(N):
            for k in range(2):
                cons.append(wall_dist[i, k] - r)
        return cons

    def constraints_jac(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[-1]
        n_cons = N * (N - 1) // 2 + 2 * N
        J = np.zeros((n_cons, 2 * N + 1))
        d = cdist(c, c)
        np.fill_diagonal(d, 1.0)
        row = 0
        for i in range(N):
            for j in range(i + 1, N):
                dij = c[i] - c[j]
                norm = d[i, j]
                if norm > 1e-12:
                    J[row, 2 * i] = dij[0] / norm
                    J[row, 2 * i + 1] = dij[1] / norm
                    J[row, 2 * j] = -dij[0] / norm
                    J[row, 2 * j + 1] = -dij[1] / norm
                J[row, -1] = -2.0
                row += 1
        for i in range(N):
            for k in range(2):
                sign = np.sign(c[i, k])
                if sign == 0:
                    sign = 1.0
                J[row, 2 * i + k] = -sign
                J[row, -1] = -1.0
                row += 1
        return J

    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]

    def fun_con(x):
        return np.array(constraints(x))

    def jac_con(x):
        return constraints_jac(x)

    cons = [{'type': 'ineq', 'fun': fun_con, 'jac': jac_con}]

    best_c = c_init.copy()
    best_r = r_init

    # Run SLSQP with increasing maxiter and tighter tolerances
    for maxiter, ftol in [(200, 1e-12), (400, 1e-14), (600, 1e-16)]:
        if (time.time() - t0) > t0 + time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                jac=objective_grad,
                bounds=bounds,
                constraints=cons,
                method='SLSQP',
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            c_new = res.x[:2 * N].reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                x0 = res.x.copy()
        except Exception:
            pass

    return best_c, best_r


def _final_local_search(c, r, max_iter=400, step_scale=1e-5):
    """Deterministic local search with adaptive step size and more directions."""
    c = c.copy()
    r_cur = _radius_fast(c)

    # More directions: 36 base + 12 diagonal
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    fine_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    diag_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12
    diag_dirs = np.column_stack([np.cos(diag_angles), np.sin(diag_angles)])
    all_dirs = np.vstack([fine_dirs, diag_dirs])

    for it in range(max_iter):
        improved = False
        for i in range(N):
            for d in all_dirs:
                for scale in [step_scale, step_scale * 0.5, step_scale * 2, step_scale * 0.25]:
                    c_try = c.copy()
                    c_try[i] += scale * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-12:
                        c = c_try
                        r_cur = r_try
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            step_scale *= 0.7
            if step_scale < 1e-10:
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
    """Gradually increase the target radius R."""
    c = start.copy()
    best_c = c.copy()
    best_r = _radius_fast(c)
    R = R0
    while R <= R1 and (time.time() - t0) < DEADLINE - 25:
        c = _relax_hard_disk(c, R, iters_per_step, eta0, eta1)
        r = _radius_fast(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        R += step
    return best_c, best_r


def _lbfgs_polish(c, t0, time_budget=25.0):
    """L-BFGS-B polish using soft-min radius with adaptive temperature."""
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    temps = [100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 1.5, 0.8]
    maxiter_per_temp = 250

    for t in temps:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget:
            break
        mi = maxiter_per_temp if t < 5.0 else maxiter_per_temp // 2
        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': mi, 'gtol': 1e-12, 'ftol': 1e-16}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
            x = res.x.copy()

    return best_x.reshape(N, 2), best_r


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
        if (time.time() - t0) > DEADLINE - 60:
            break
        # Phase 1: hard-disk radius expansion
        c, r = _grow_radius(start, 0.04, 0.072, 0.0005, 500, 0.02, 0.002, t0)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = inits[0]
        best_r = _radius_fast(best_c)

    # Phase 2: SLSQP Epigraph Polish (primary high-precision method)
    if (time.time() - t0) < DEADLINE - 30:
        c_slsqp, r_slsqp = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(35.0, DEADLINE - 30 - (time.time() - t0))
        )
        if r_slsqp > best_r:
            best_r = r_slsqp
            best_c = c_slsqp

    # Phase 3: Voronoi-style refinement
    if (time.time() - t0) < DEADLINE - 25:
        c_vor, r_vor = _voronoi_style_refine(
            best_c, best_r, t0,
            time_budget=min(25.0, DEADLINE - 25 - (time.time() - t0))
        )
        if r_vor > best_r:
            best_r = r_vor
            best_c = c_vor

    # Phase 4: L-BFGS-B soft-min polish
    if (time.time() - t0) < DEADLINE - 15:
        c_polished, r_polished = _lbfgs_polish(
            best_c, t0,
            time_budget=min(25.0, DEADLINE - 15 - (time.time() - t0))
        )
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 5: Another SLSQP pass if time allows
    if (time.time() - t0) < DEADLINE - 10:
        c_slsqp2, r_slsqp2 = _slsqp_epigraph_polish(
            best_c, best_r, t0,
            time_budget=min(20.0, DEADLINE - 10 - (time.time() - t0))
        )
        if r_slsqp2 > best_r:
            best_r = r_slsqp2
            best_c = c_slsqp2

    # Phase 6: Final high-precision local search
    if (time.time() - t0) < DEADLINE - 5:
        c_ls, r_ls = _final_local_search(best_c, best_r, max_iter=500, step_scale=1e-5)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

        # One more round with even smaller steps
        if (time.time() - t0) < DEADLINE - 2:
            c_ls2, r_ls2 = _final_local_search(best_c, best_r, max_iter=400, step_scale=3e-6)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2

    # Final validation: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)

    return best_c
