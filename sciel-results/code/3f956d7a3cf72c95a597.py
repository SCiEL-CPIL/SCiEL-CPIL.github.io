"""Combined packing program: SLSQP linear-epigraph primary + multi-stage refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

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


def _force_relax(c, r, time_budget=3.0, max_iter=15000):
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


def _langevin_dynamics(c0, r, time_budget=6.0, max_iter=100000, rng=None):
    """Langevin dynamics for hard-disk packing at fixed radius r."""
    if rng is None:
        rng = np.random.default_rng(42)
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    v = np.zeros_like(c)
    start = time.time()
    dt = 0.002
    gamma = 0.1
    T_init = 0.01
    T_min = 0.00001
    n = max_iter
    best_c = c.copy()
    best_score = _true_score(c)
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        T = T_init * (T_min / T_init) ** (i / max(1, n))
        disp = np.zeros_like(c)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        if len(pairs) > 0:
            for idx in pairs:
                i, j = int(idx[0]), int(idx[1])
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
        noise = rng.normal(0, 1, size=c.shape)
        v = (v - dt * disp) * np.exp(-gamma * dt) + np.sqrt(2 * T * dt * gamma) * noise
        c_new = c + dt * v
        c_new = np.clip(c_new, -0.5 + r, 0.5 - r)
        tree_new = cKDTree(c_new)
        pairs_new = tree_new.query_pairs(2 * r - 1e-9, output_type='ndarray')
        feasible = len(pairs_new) == 0
        if feasible:
            score = _true_score(c_new)
            if score >= r - 1e-9:
                c = c_new
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
            else:
                if np.random.random() < np.exp((score - r) / max(T, 1e-12)):
                    c = c_new
        else:
            v *= 0.5
    return best_c


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish with adaptive k schedule for tighter approximation."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start_time = time.time()
    best_c = c0.copy()
    best_r = _true_score(c0)

    def obj_and_grad(x, kk):
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
        weights = np.exp(-kk * all_d)
        sw = weights.sum()
        val = -np.log(sw) / kk
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

    # Adaptive k schedule: start moderate, ramp up for tighter min approximation
    k_schedule = [k, k * 1.5, k * 2.0]
    for kk in k_schedule:
        if time.time() - start_time > time_budget * 0.9:
            break
        try:
            res = minimize(lambda x: obj_and_grad(x, kk), c0.ravel(), jac=True,
                           method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 120, 'maxfun': 3000,
                                    'ftol': 1e-14, 'gtol': 1e-10,
                                    'iprint': 0})
            c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
            s = _true_score(c)
            if s > best_r:
                best_r = s
                best_c = c.copy()
            c0 = c  # warm-start next iteration
        except Exception:
            break

    return best_c


def _slsqp_linear_epigraph(c0, r_init, time_budget=15.0, max_iter=400, ftol=1e-13):
    """SLSQP with LINEAR epigraph formulation (vectorized constraints):
    maximize t subject to:
      t <= |c_i - c_j| / 2  for all i < j
      t <= 0.5 - c[i,k]     for all i, k (upper walls)
      t <= c[i,k] + 0.5     for all i, k (lower walls)
    Vectorized for speed."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)

    # Precompute pair indices
    ii, jj = np.triu_indices(N, k=1)
    n_pairs = len(ii)

    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    def pair_con(x):
        pts = x[:2 * N].reshape(N, 2)
        ci = pts[ii]
        cj = pts[jj]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
        return d / 2.0 - x[2 * N]

    def pair_con_grad(x):
        pts = x[:2 * N].reshape(N, 2)
        ci = pts[ii]
        cj = pts[jj]
        diff = ci - cj
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        safe_d = np.where(d > 1e-12, d, 1e-12)
        g = np.zeros(2 * N + 1)
        # Scatter pair gradients
        idx_i = ii * 2
        idx_j = jj * 2
        factor = 1.0 / (2.0 * safe_d)
        np.add.at(g, idx_i, factor * diff[:, 0])
        np.add.at(g, idx_i + 1, factor * diff[:, 1])
        np.add.at(g, idx_j, -factor * diff[:, 0])
        np.add.at(g, idx_j + 1, -factor * diff[:, 1])
        g[2 * N] = -1.0
        return g

    def wall_con(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        # 0.5 - c[i,0] - t, c[i,0]+0.5 - t, 0.5 - c[i,1] - t, c[i,1]+0.5 - t
        return np.concatenate([
            0.5 - pts[:, 0] - t,
            pts[:, 0] + 0.5 - t,
            0.5 - pts[:, 1] - t,
            pts[:, 1] + 0.5 - t,
        ])

    def wall_con_grad(x):
        g = np.zeros(2 * N + 1)
        g[0::2] = -1.0   # 0.5 - c[i,0]
        g[1::2] = 1.0    # c[i,0]+0.5
        # For y: 0.5 - c[i,1] has coeff -1 at index 2i+1, c[i,1]+0.5 has +1
        # But we need to be careful: g[0::2] gives indices 0,2,4,... which are x-coords
        # g[1::2] gives indices 1,3,5,... which are y-coords
        # Actually let me redo: for each point i, x-coord is at 2i, y-coord at 2i+1
        # wall_con returns: [0.5-x0-t, x0+0.5-t, 0.5-x1-t, x1+0.5-t, ...] for each i
        # So gradient w.r.t. x_i (index 2i) is -1 then +1 for the two x-walls
        # Gradient w.r.t. y_i (index 2i+1) is -1 then +1 for the two y-walls
        # The t gradient is -1 for all 4N wall constraints
        g = np.zeros(2 * N + 1)
        # For x-coords: -1 for "0.5 - x - t", +1 for "x + 0.5 - t"
        g[0::2] = 0.0  # will be set below properly
        # Actually simpler: just build it
        g = np.zeros(2 * N + 1)
        for i in range(N):
            g[2*i] = 0.0   # will accumulate
            g[2*i+1] = 0.0
        # x-walls: 0.5 - x_i - t (grad -1 at 2i), x_i + 0.5 - t (grad +1 at 2i)
        # They cancel! Net gradient on x_i from x-walls is 0.
        # Same for y_i from y-walls is 0.
        # Only t gets -4N contribution
        g[2 * N] = -4.0 * N
        return g

    cons = [
        {'type': 'ineq', 'fun': pair_con, 'jac': pair_con_grad},
        {'type': 'ineq', 'fun': wall_con, 'jac': wall_con_grad},
    ]

    x0 = np.concatenate([c0.ravel(), [r_init]])
    start = time.time()
    best_c = c0.copy()
    best_r = r_init

    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                       constraints=cons,
                       options={'maxiter': max_iter, 'ftol': ftol, 'disp': False})
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_res = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
            s = _true_score(c_res)
            if s > best_r:
                best_r, best_c = s, c_res
    except Exception:
        pass

    # Second pass: warm-start from the result with tighter tolerance
    if time.time() - start < time_budget * 0.7 and best_r > r_init:
        try:
            x0b = np.concatenate([best_c.ravel(), [best_r]])
            res2 = minimize(neg_t, x0b, jac=neg_t_grad, method='SLSQP',
                           bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                           constraints=cons,
                           options={'maxiter': max_iter, 'ftol': ftol * 0.01, 'disp': False})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                c_res2 = np.clip(res2.x[:2 * N].reshape(N, 2), -0.5, 0.5)
                s2 = _true_score(c_res2)
                if s2 > best_r:
                    best_r, best_c = s2, c_res2
        except Exception:
            pass

    return best_c


def _delaunay_active_set_refine(c0, time_budget=5.0, max_iter=200):
    """Delaunay-based active-set refinement: identify near-contact pairs via
    Delaunay triangulation and optimize only those constraints plus walls."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5, 0.5)
    start = time.time()
    best_c = c.copy()
    best_r = _true_score(c)

    for outer in range(max_iter):
        if time.time() - start > time_budget:
            break
        # Compute current radius
        r_cur = _true_score(c)
        if r_cur <= best_r:
            break

        # Find near-contact pairs using Delaunay + distance check
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(c)
            # Get unique edges from triangulation
            edges_set = set()
            for simplex in tri.simplices:
                for k in range(3):
                    a, b = int(simplex[k]), int(simplex[(k + 1) % 3])
                    edges_set.add((min(a, b), max(a, b)))
            edges = list(edges_set)
        except Exception:
            edges = []

        # Also include pairs within 2.5*r_cur of each other
        if not edges:
            d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            thresh = 2.5 * r_cur
            edges = [(int(i), int(j)) for i, j in np.argwhere(d < thresh) if i < j]

        # Build active constraint set: near-contact pairs + active walls
        active_pairs = []
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2.2 * r_cur:
                active_pairs.append((i, j))

        # Limit to top 150 closest pairs if too many
        if len(active_pairs) > 150:
            d_arr = np.array([np.linalg.norm(c[i] - c[j]) for i, j in active_pairs])
            idx = np.argsort(d_arr)[:150]
            active_pairs = [active_pairs[k] for k in idx]

        active_walls = []
        for k in range(N):
            for axis in range(2):
                dist = 0.5 - c[k, axis]
                if dist < 1.1 * r_cur:
                    active_walls.append((k, axis, 1))
                dist = c[k, axis] + 0.5
                if dist < 1.1 * r_cur:
                    active_walls.append((k, axis, -1))

        if len(active_pairs) == 0 and len(active_walls) == 0:
            break

        # Build SLSQP with only active constraints
        def neg_t(x):
            return -x[2 * N]

        def neg_t_grad(x):
            g = np.zeros(2 * N + 1)
            g[2 * N] = -1.0
            return g

        cons = []
        for i, j in active_pairs:
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
                diff = ci - cj
                d = np.sqrt(np.sum(diff ** 2))
                if d > 1e-12:
                    g[2 * i:2 * i + 2] = diff / (2 * d)
                    g[2 * j:2 * j + 2] = -diff / (2 * d)
                g[2 * N] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_con_grad})

        for k, axis, sign in active_walls:
            if sign > 0:
                def wall_con(x, k=k, axis=axis):
                    return 0.5 - x[2 * k + axis] - x[2 * N]
                def wall_grad(x, k=k, axis=axis):
                    g = np.zeros(2 * N + 1)
                    g[2 * k + axis] = -1.0
                    g[2 * N] = -1.0
                    return g
            else:
                def wall_con(x, k=k, axis=axis):
                    return x[2 * k + axis] + 0.5 - x[2 * N]
                def wall_grad(x, k=k, axis=axis):
                    g = np.zeros(2 * N + 1)
                    g[2 * k + axis] = 1.0
                    g[2 * N] = -1.0
                    return g
            cons.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_grad})

        x0 = np.concatenate([c.ravel(), [r_cur]])
        try:
            res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                           bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                           constraints=cons,
                           options={'maxiter': 80, 'ftol': 1e-14, 'disp': False})
            if res.x is not None and np.all(np.isfinite(res.x)):
                c_new = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
                s = _true_score(c_new)
                if s > best_r:
                    best_r = s
                    best_c = c_new.copy()
                    c = c_new
                else:
                    break
        except Exception:
            break

    return best_c





def _contact_graph_refine(c0, r_target, time_budget=3.0, max_iter=300):
    """Contact-graph-aware local optimization."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
    start = time.time()
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        wall_dists = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)
        min_pair = d.min()
        min_wall = wall_dists.min()
        tol = 1e-4
        active_pairs = np.argwhere(d < min_pair + tol)
        active_pairs = [(int(i), int(j)) for i, j in active_pairs if i < j]
        active_walls = []
        for k in range(N):
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    dist = 0.5 - c[k, axis]
                else:
                    dist = c[k, axis] + 0.5
                if dist < min_wall + tol:
                    active_walls.append((k, axis, sign))
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        n_vars = 2 * N
        A = []
        for i, j in active_pairs:
            diff = c[i] - c[j]
            row = np.zeros(n_vars)
            row[2 * i:2 * i + 2] = diff
            row[2 * j:2 * j + 2] = -diff
            A.append(row)
        for k, axis, sign in active_walls:
            row = np.zeros(n_vars)
            row[2 * k + axis] = sign
            A.append(row)
        if len(A) == 0:
            break
        A = np.array(A)
        disp = np.zeros(n_vars)
        for row in A:
            disp += row
        norm = np.linalg.norm(disp)
        if norm < 1e-12:
            break
        disp = disp / norm
        best_c = c.copy()
        best_r = _true_score(c)
        for step in [0.001, 0.005, 0.01, 0.02, 0.05]:
            c_try = c + step * disp.reshape(N, 2)
            c_try = np.clip(c_try, -0.5 + r_target * 0.9, 0.5 - r_target * 0.9)
            s = _true_score(c_try)
            if s > best_r:
                best_r = s
                best_c = c_try.copy()
        c = best_c
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
        if time.time() - start_time > DEADLINE - 40:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 40:
                break
            cur = _force_relax(cur, r, time_budget=1.0, max_iter=2000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.06
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
    if time.time() - start_time < DEADLINE - 35:
        ld_budget = min(8.0, DEADLINE - 35 - (time.time() - start_time))
        if ld_budget > 3.0:
            rng_ld = np.random.default_rng(777)
            cur = _langevin_dynamics(best, best_r, time_budget=ld_budget, max_iter=80000, rng=rng_ld)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 3) Soft-min polish with increasing k (replaces slow global SLSQP)
    if time.time() - start_time < DEADLINE - 20:
        c = _softmin_refine(best, k=100.0, time_budget=5.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 4) Delaunay active-set refinement (sparse, fast, exact)
    if time.time() - start_time < DEADLINE - 15:
        das_budget = min(10.0, DEADLINE - 15 - (time.time() - start_time))
        if das_budget > 3.0:
            cur = _delaunay_active_set_refine(best, time_budget=das_budget, max_iter=300)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 5) High-k Soft-min polish
    if time.time() - start_time < DEADLINE - 10:
        c = _softmin_refine(best, k=500.0, time_budget=8.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 6) Final Delaunay active-set refinement
    if time.time() - start_time < DEADLINE - 5:
        das_budget = min(8.0, DEADLINE - 5 - (time.time() - start_time))
        if das_budget > 3.0:
            cur = _delaunay_active_set_refine(best, time_budget=das_budget, max_iter=300)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 7) Final tiny soft-min polish
    if time.time() - start_time < DEADLINE - 1:
        c = _softmin_refine(best, k=1000.0, time_budget=2.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
