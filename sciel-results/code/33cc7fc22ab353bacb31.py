"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])  # 49 points
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _force_relax(c, r, time_budget=4.0, max_iter=20000):
    """Grow/relax a hard-disk packing: push overlapping circles apart and
    keep them inside the square. Returns a feasible layout for radius r."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        # pairwise overlap repulsion
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            # unit vectors from j to i
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r - d) / (2 * r)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        # wall push
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        # Adaptive timestep: reduce if large displacements
        max_disp = np.max(np.sqrt((disp ** 2).sum(axis=1)))
        if max_disp > 0.01:
            dt_eff = dt * min(1.0, 0.01 / max_disp)
        else:
            dt_eff = dt
        c = np.clip(c + dt_eff * disp, -0.5 + r, 0.5 - r)
    return c


def _slsqp_epigraph(c0, r0, time_budget=6.0, maxiter=300):
    """SLSQP: maximize r s.t. ||c_i-c_j||^2 >= 4r^2, wall constraints.
    Variables x = [c_0..c_{N-1}, r] (2N+1)."""
    x0 = np.empty(2 * N + 1)
    x0[:2 * N] = np.asarray(c0, dtype=float).ravel()
    x0[-1] = r0
    x0 = np.clip(x0[:2 * N], -0.5, 0.5).reshape(-1)
    x0 = np.concatenate([x0, [r0]])

    def obj(x):
        return -x[-1]

    def obj_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    # Pairwise constraints: (c_i - c_j)^2 >= 4 r^2  =>  g = 4r^2 - ||c_i-c_j||^2 <= 0
    pairs = np.argwhere(np.triu(np.ones((N, N), dtype=bool), k=1))
    pi, pj = pairs[:, 0], pairs[:, 1]
    n_pairs = len(pi)

    def con_pairs(x):
        r = x[-1]
        ci = x[2 * pi: 2 * pi + 2].reshape(-1, 2)
        cj = x[2 * pj: 2 * pj + 2].reshape(-1, 2)
        d2 = ((ci - cj) ** 2).sum(axis=1)
        return 4 * r * r - d2

    def con_pairs_jac(x):
        r = x[-1]
        ci = x[2 * pi: 2 * pi + 2].reshape(-1, 2)
        cj = x[2 * pj: 2 * pj + 2].reshape(-1, 2)
        diff = ci - cj
        J = np.zeros((n_pairs, 2 * N + 1))
        # d(4r^2 - d2)/d c_i = -2 * diff
        # d(4r^2 - d2)/d c_j =  2 * diff
        # d(4r^2 - d2)/d r   =  8r
        J[np.arange(n_pairs), 2 * pi] = -2 * diff[:, 0]
        J[np.arange(n_pairs), 2 * pi + 1] = -2 * diff[:, 1]
        J[np.arange(n_pairs), 2 * pj] = 2 * diff[:, 0]
        J[np.arange(n_pairs), 2 * pj + 1] = 2 * diff[:, 1]
        J[:, -1] = 8 * r
        return J

    # Wall constraints: c_ik >= -0.5 + r  =>  -c_ik + r - 0.5 <= 0
    #                    c_ik <=  0.5 - r  =>   c_ik + r + 0.5 <= 0
    def con_walls(x):
        r = x[-1]
        cx = x[0::2]
        cy = x[1::2]
        # -cx + r - 0.5 <= 0
        #  cx + r + 0.5 <= 0
        # -cy + r - 0.5 <= 0
        #  cy + r + 0.5 <= 0
        return np.concatenate([
            -cx + r - 0.5,
            cx + r + 0.5,
            -cy + r - 0.5,
            cy + r + 0.5,
        ])

    def con_walls_jac(x):
        J = np.zeros((4 * N, 2 * N + 1))
        # row 0..N-1: -cx + r - 0.5
        J[np.arange(N), 2 * np.arange(N)] = -1.0
        J[np.arange(N), -1] = 1.0
        # row N..2N-1: cx + r + 0.5
        J[N + np.arange(N), 2 * np.arange(N)] = 1.0
        J[N + np.arange(N), -1] = 1.0
        # row 2N..3N-1: -cy + r - 0.5
        J[2 * N + np.arange(N), 2 * np.arange(N) + 1] = -1.0
        J[2 * N + np.arange(N), -1] = 1.0
        # row 3N..4N-1: cy + r + 0.5
        J[3 * N + np.arange(N), 2 * np.arange(N) + 1] = 1.0
        J[3 * N + np.arange(N), -1] = 1.0
        return J

    cons = [
        {'type': 'ineq', 'fun': lambda x: -con_pairs(x), 'jac': lambda x: -con_pairs_jac(x)},
        {'type': 'ineq', 'fun': lambda x: -con_walls(x), 'jac': lambda x: -con_walls_jac(x)},
    ]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.01, 0.15)]

    start = time.time()
    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP', bounds=bounds,
                       constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': 0})
        if res.success or res.fun < obj(x0):
            c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
            r = float(res.x[-1])
            # Verify feasibility
            s = _true_score(c)
            if s > 0:
                return c
    except Exception:
        pass
    return None


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        # Use log-sum-exp for numerical stability
        m = np.min(all_d)
        log_sw = m + np.log(np.sum(np.exp(k * (m - all_d))))
        val = log_sw / k

        weights = np.exp(-k * all_d)
        sw = weights.sum()
        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 500, 'maxfun': 12000,
                                'ftol': 1e-18, 'gtol': 1e-14,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _langevin_optimize(c0, r, time_budget=5.0, max_iter=50000, noise_scale=0.002):
    """Langevin dynamics: gradient descent on soft-min with thermal noise."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5, 0.5)
    start = time.time()
    k = 200.0  # temperature parameter for soft-min

    def soft_min_grad(pts):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return grad_pts

    lr = 0.001
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        grad = soft_min_grad(c)
        noise = np.random.normal(0, noise_scale, c.shape)
        c = c - lr * grad + noise
        c = np.clip(c, -0.5, 0.5)
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(12):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        # estimate a starting radius from the layout
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(12):
            if time.time() - start_time > DEADLINE - 20:
                break
            cur = _force_relax(cur, r, time_budget=2.0, max_iter=5000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.03
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Langevin optimization phase for better exploration
    if time.time() - start_time < DEADLINE - 25:
        for c0 in [best] + inits[:3]:
            if time.time() - start_time > DEADLINE - 25:
                break
            c = _langevin_optimize(c0, best_r, time_budget=4.0, max_iter=30000, noise_scale=0.0015)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 3) SLSQP epigraph refinement (handles active-set structure directly)
    if time.time() - start_time < DEADLINE - 25:
        c = _slsqp_epigraph(best, best_r, time_budget=8.0, maxiter=200)
        if c is not None:
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 4) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.5, max_iter=2000))
    for c0 in inits[:4]:
        candidates.append(_force_relax(c0, best_r, time_budget=1.5, max_iter=2000))

    # Progressive k values for better convergence
    for k in (50.0, 100.0, 200.0, 400.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 15:
                break
            c = _softmin_refine(c0, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
                # Update candidates with new best
                candidates[0] = best
                candidates[1] = _force_relax(best, best_r, time_budget=1.0, max_iter=1000)

    # 5) Second SLSQP pass with higher maxiter for more precise convergence
    if time.time() - start_time < DEADLINE - 12:
        c = _slsqp_epigraph(best, best_r, time_budget=10.0, maxiter=400)
        if c is not None:
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 6) Final high-precision polish on the best layout with tighter tolerance
    if time.time() - start_time < DEADLINE - 5:
        for k in (400.0, 800.0, 1200.0):
            if time.time() - start_time > DEADLINE - 5:
                break
            c = _softmin_refine(best, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 7) Very fine-tuning with very high k
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 1500.0, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
