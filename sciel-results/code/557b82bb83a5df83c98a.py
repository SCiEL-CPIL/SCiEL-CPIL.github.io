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
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

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
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k

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
                       options={'maxiter': 120, 'maxfun': 3000,
                                'ftol': 1e-13, 'gtol': 1e-9,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _slsqp_epigraph(c0, r_init, time_budget=5.0):
    """
    SLSQP with epigraph formulation: maximize t subject to
      t <= |c_i - c_j| / 2  for all i < j
      t <= 0.5 - |c_ik|     for all i, k
    This directly optimizes the true max-min radius.
    """
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)
    
    def neg_t(x):
        """Objective: -t (we minimize -t to maximize t)."""
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        return -t

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    def _make_constraints():
        cons = []
        # Pairwise distance constraints: |c_i - c_j| / 2 - t >= 0
        for i in range(N):
            for j in range(i + 1, N):
                def pair_con(x, i=i, j=j):
                    ci = x[2 * i:2 * i + 2]
                    cj = x[2 * j:2 * j + 2]
                    t = x[2 * N]
                    d = np.sqrt(np.sum((ci - cj) ** 2))
                    return d / 2.0 - t
                def pair_con_grad(x, i=i, j=j):
                    g = np.zeros(2 * N + 1)
                    ci = x[2 * i:2 * i + 2]
                    cj = x[2 * j:2 * j + 2]
                    t = x[2 * N]
                    diff = ci - cj
                    d = np.sqrt(np.sum(diff ** 2))
                    if d > 1e-12:
                        g[2 * i:2 * i + 2] = diff / (2 * d)
                        g[2 * j:2 * j + 2] = -diff / (2 * d)
                    g[2 * N] = -1.0
                    return g
                cons.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_con_grad})
        
        # Wall constraints: 0.5 - |c_ik| - t >= 0
        for i in range(N):
            for axis in range(2):
                # 0.5 - c[i, axis] - t >= 0
                def wall_con(x, i=i, axis=axis, sign=1):
                    t = x[2 * N]
                    return 0.5 - x[2 * i + axis] - t
                def wall_con_grad(x, i=i, axis=axis, sign=1):
                    g = np.zeros(2 * N + 1)
                    g[2 * i + axis] = -1.0
                    g[2 * N] = -1.0
                    return g
                cons.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_con_grad})
                
                # c[i, axis] + 0.5 - t >= 0
                def wall_con2(x, i=i, axis=axis, sign=-1):
                    t = x[2 * N]
                    return x[2 * i + axis] + 0.5 - t
                def wall_con2_grad(x, i=i, axis=axis, sign=-1):
                    g = np.zeros(2 * N + 1)
                    g[2 * i + axis] = 1.0
                    g[2 * N] = -1.0
                    return g
                cons.append({'type': 'ineq', 'fun': wall_con2, 'jac': wall_con2_grad})
        
        return cons

    cons = _make_constraints()
    
    x0 = np.concatenate([c0.ravel(), [r_init]])
    
    start = time.time()
    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                       constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-12,
                                'disp': False})
        c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    
    if time.time() - start > time_budget:
        return c
    return c


def _langevin_optimize(c0, r_target, time_budget=6.0):
    """
    Langevin dynamics with cooling to find a dense packing.
    """
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
    
    start = time.time()
    dt = 0.002
    gamma = 0.1
    T = 0.01
    T_min = 0.0001
    cool_rate = 0.999
    
    max_steps = 100000
    
    for step in range(max_steps):
        if time.time() - start > time_budget:
            break
        
        forces = np.zeros_like(c)
        
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        close = d < 3 * r_target
        if np.any(close):
            safe_d = np.where(d > 1e-10, d, 1e-10)
            overlap = r_target - d
            mask = overlap > 0
            if np.any(mask):
                f_mag = 10.0 * overlap
                vec = (c[:, None, :] - c[None, :, :]) / safe_d[:, :, None]
                f_vec = np.where(mask[:, :, None], f_mag[:, :, None] * vec, 0.0)
                forces += f_vec.sum(axis=1)
        
        margin = 0.01
        for axis in [0, 1]:
            dist_r = 0.5 - c[:, axis]
            pen_r = np.clip(margin - dist_r, 0, None)
            forces[:, axis] += 10.0 * pen_r
            
            dist_l = c[:, axis] + 0.5
            pen_l = np.clip(margin - dist_l, 0, None)
            forces[:, axis] -= 10.0 * pen_l
        
        noise = np.random.normal(0, np.sqrt(2 * T * dt), c.shape)
        c = c + dt * (forces + noise) / (1 + dt * gamma)
        
        c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
        
        T *= cool_rate
        if T < T_min:
            T = T_min
    
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(8):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 15:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 15:
                break
            cur = _force_relax(cur, r, time_budget=2.0, max_iter=4000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.05
                hi = r
            else:
                hi = r
                if hi - lo < 1e-6:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Langevin optimization from best and a few others
    for c0 in [best] + inits[:3]:
        if time.time() - start_time > DEADLINE - 10:
            break
        cur = _langevin_optimize(c0, best_r, time_budget=3.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 3) SLSQP epigraph optimization - directly maximizes the true radius
    if time.time() - start_time < DEADLINE - 12:
        # Run SLSQP from the current best
        cur = _slsqp_epigraph(best, best_r, time_budget=6.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur
    
    # 4) More SLSQP from other starts if time permits
    for c0 in inits[:4]:
        if time.time() - start_time > DEADLINE - 10:
            break
        cur = _slsqp_epigraph(c0, best_r * 0.95, time_budget=4.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 5) Smooth soft-min polish from the best layout
    candidates = [best]
    if time.time() - start_time < DEADLINE - 8:
        candidates.append(_force_relax(best, best_r, time_budget=1.5, max_iter=2000))
    for c0 in inits[:3]:
        if time.time() - start_time > DEADLINE - 8:
            break
        candidates.append(_force_relax(c0, best_r, time_budget=1.5, max_iter=2000))

    for k in (80.0, 150.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 6:
                break
            c = _softmin_refine(c0, k, time_budget=2.5)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 6) Final SLSQP polish at the best radius found
    if time.time() - start_time < DEADLINE - 4:
        cur = _slsqp_epigraph(best, best_r, time_budget=4.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 7) Final soft-min polish
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 200.0, time_budget=2.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
