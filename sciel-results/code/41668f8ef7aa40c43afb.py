"""50 equal circles in a unit square: Langevin + soft-min + contact-graph refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
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
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-9, output_type='ndarray')
        if len(pairs) > 0:
            for idx in pairs:
                i, j = idx[0], idx[1]
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[i] += force
                    disp[j] -= force
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


def _langevin_dynamics(c0, r, time_budget=8.0, max_iter=200000, rng=None):
    """Langevin dynamics for hard-disk packing at fixed radius r."""
    if rng is None:
        rng = np.random.default_rng(42)
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    v = np.zeros_like(c)
    start = time.time()
    dt = 0.003
    gamma = 0.15
    T_init = 0.015
    T_min = 0.000005
    n = max(1, max_iter)
    best_c = c.copy()
    best_score = _true_score(c)
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        T = T_init * (T_min / T_init) ** (i / n)
        disp = np.zeros_like(c)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        if len(pairs) > 0:
            for idx in pairs:
                ii, jj = idx[0], idx[1]
                diff = c[ii] - c[jj]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[ii] += force
                    disp[jj] -= force
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        noise = rng.normal(0, 1, size=c.shape)
        v = (v - dt * disp) * np.exp(-gamma * dt) + np.sqrt(2 * T * dt * gamma) * noise
        c_new = c + dt * v
        c_new = np.clip(c_new, -0.5 + r, 0.5 - r)
        feasible = True
        tree_new = cKDTree(c_new)
        pairs_new = tree_new.query_pairs(2 * r - 1e-9, output_type='ndarray')
        if len(pairs_new) > 0:
            feasible = False
        if feasible:
            score = _true_score(c_new)
            if score >= r - 1e-9:
                c = c_new
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
            else:
                if rng.random() < np.exp((score - r) / max(T, 1e-12)):
                    c = c_new
        else:
            v *= 0.5
    return best_c


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
                       options={'maxiter': 500, 'maxfun': 15000,
                                'ftol': 1e-18, 'gtol': 1e-12,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _contact_graph_refine(c0, r, time_budget=6.0):
    """Refine using Delaunay triangulation to identify contact graph,
    then solve for equilibrium with exact distance constraints."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    for iteration in range(30):
        if time.time() - start > time_budget:
            break
        try:
            tri = Delaunay(c)
        except Exception:
            break
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        wall_contacts = {}
        for k in range(N):
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    dist = 0.5 - c[k, axis]
                else:
                    dist = c[k, axis] + 0.5
                if dist < r + 1e-6:
                    if k not in wall_contacts:
                        wall_contacts[k] = []
                    wall_contacts[k].append((axis, sign))
        disp = np.zeros_like(c)
        for i, j in edges:
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                continue
            target = 2 * r
            if d < target:
                correction = (target - d) / d * 0.5
                disp[i] += correction * diff
                disp[j] -= correction * diff
        for k, walls in wall_contacts.items():
            for axis, sign in walls:
                if sign > 0:
                    pen = c[k, axis] - (0.5 - r)
                else:
                    pen = (-0.5 + r) - c[k, axis]
                if pen > 0:
                    disp[k, axis] -= sign * pen
        c = np.clip(c + 0.1 * disp, -0.5 + r, 0.5 - r)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        w = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)
        min_pair = d.min()
        min_wall = w.min()
        if min_pair >= 2 * r - 1e-10 and min_wall >= r - 1e-10:
            break
    return c


def _slsqp_epigraph(c0, r0, time_budget=10.0):
    """SLSQP with epigraph formulation: maximize t subject to
    pairwise distances >= 2t and wall distances >= t."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r0, 0.01)

    def neg_t(x):
        """Minimize -t where x = [c_1x, c_1y, ..., c_Nx, c_Ny, t]."""
        t = x[-1]
        return -t

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g

    # Constraints: pairwise distances - 2t >= 0
    def pairwise_constraint(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[-1]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        # Return all pairwise constraints: d_ij - 2t >= 0
        return (d - 2 * t).ravel()

    def pairwise_jacobian(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[-1]
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, 1.0)
        # Jacobian of d_ij - 2t w.r.t. x
        # d/dx_k of d_ij = (pts_i - pts_j) / d_ij
        jac = np.zeros((N * N, 2 * N + 1))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                row = i * N + j
                diff_ij = pts[i] - pts[j]
                d_ij = dists[i, j]
                if d_ij > 1e-12:
                    jac[row, 2 * i:2 * i + 2] = diff_ij / d_ij
                    jac[row, 2 * j:2 * j + 2] = -diff_ij / d_ij
                jac[row, -1] = -2.0
        return jac

    # Constraints: wall distances - t >= 0
    def wall_constraint(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[-1]
        walls = np.stack([
            0.5 - pts[:, 0], pts[:, 0] + 0.5,
            0.5 - pts[:, 1], pts[:, 1] + 0.5,
        ], axis=0)  # shape (4, N)
        return (walls - t).ravel()

    def wall_jacobian(x):
        jac = np.zeros((4 * N, 2 * N + 1))
        # For each wall constraint: d_wall - t >= 0
        # d/dx_k of wall distance is 0 or ±1
        for k in range(N):
            # wall 0: 0.5 - x_kx >= t  ->  jac = [-1, 0, ..., -1]
            jac[4 * k, 2 * k] = -1.0
            jac[4 * k, -1] = -1.0
            # wall 1: x_kx + 0.5 >= t  ->  jac = [1, 0, ..., -1]
            jac[4 * k + 1, 2 * k] = 1.0
            jac[4 * k + 1, -1] = -1.0
            # wall 2: 0.5 - x_ky >= t  ->  jac = [0, -1, ..., -1]
            jac[4 * k + 2, 2 * k + 1] = -1.0
            jac[4 * k + 2, -1] = -1.0
            # wall 3: x_ky + 0.5 >= t  ->  jac = [0, 1, ..., -1]
            jac[4 * k + 3, 2 * k + 1] = 1.0
            jac[4 * k + 3, -1] = -1.0
        return jac

    x0 = np.concatenate([c0.ravel(), [r0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]

    cons = [
        {'type': 'ineq', 'fun': pairwise_constraint, 'jac': pairwise_jacobian},
        {'type': 'ineq', 'fun': wall_constraint, 'jac': wall_jacobian},
    ]

    start = time.time()
    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-12, 'iprint': 0})
        c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(10):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(12):
            if time.time() - start_time > DEADLINE - 30:
                break
            cur = _force_relax(cur, r, time_budget=1.0, max_iter=4000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.04
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Langevin dynamics to escape local minima at current best radius
    if time.time() - start_time < DEADLINE - 25:
        ld_budget = min(12.0, DEADLINE - 25 - (time.time() - start_time))
        if ld_budget > 3.0:
            rng_ld = np.random.default_rng(777)
            cur = _langevin_dynamics(best, best_r, time_budget=ld_budget, max_iter=150000, rng=rng_ld)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
            if time.time() - start_time < DEADLINE - 20 and ld_budget > 6.0:
                rng_ld2 = np.random.default_rng(999)
                perturbed = best + rng_ld2.normal(0, 0.002, size=best.shape)
                perturbed = np.clip(perturbed, -0.5 + best_r, 0.5 - best_r)
                cur2 = _langevin_dynamics(perturbed, best_r, time_budget=min(5.0, DEADLINE - 20 - (time.time() - start_time)), max_iter=80000, rng=rng_ld2)
                s2 = _true_score(cur2)
                if s2 > best_r:
                    best_r, best = s2, cur2

    # 3) Contact-graph-based refinement
    if time.time() - start_time < DEADLINE - 18:
        dc_budget = min(6.0, DEADLINE - 18 - (time.time() - start_time))
        if dc_budget > 1.0:
            cur = _contact_graph_refine(best, best_r, time_budget=dc_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 4) SLSQP epigraph formulation (direct max-min radius optimization)
    if time.time() - start_time < DEADLINE - 12:
        sl_budget = min(10.0, DEADLINE - 12 - (time.time() - start_time))
        if sl_budget > 3.0:
            cur = _slsqp_epigraph(best, best_r, time_budget=sl_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 5) Smooth soft-min polish from the best layout and a couple others.
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=2000))
    for c0 in inits[:4]:
        if time.time() - start_time > DEADLINE - 12:
            break
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=2000))

    for k in (60.0, 120.0, 200.0, 350.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 6:
                break
            c = _softmin_refine(c0, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
                candidates[0] = best

    # 6) Final high-precision polish
    if time.time() - start_time < DEADLINE - 3:
        c = _softmin_refine(best, 500.0, time_budget=5.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
        if time.time() - start_time < DEADLINE - 1:
            c = _softmin_refine(best, 800.0, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
