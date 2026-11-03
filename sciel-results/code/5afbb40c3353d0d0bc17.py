"""50-circle packing in unit square: sparse Delaunay SLSQP + active-set + Newton polish."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 118.0


def _true_score(c):
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = float(np.min(w))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _init_grid():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _init_hex():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / m - 0.5
            if i % 2 == 1:
                x += 0.5 / m
            pts.append((x, y))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _init_random(rng, n=10):
    inits = []
    for _ in range(n):
        c = rng.uniform(-0.45, 0.45, (N, 2))
        inits.append(c)
    return inits


def _force_relax(c, r, time_budget=2.0, max_iter=8000):
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.03
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
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
                    f = overlap * diff / d
                    disp[i] += f
                    disp[j] -= f
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = np.clip(c[:, axis] - (0.5 - r), 0, None)
            else:
                pen = np.clip((-0.5 + r) - c[:, axis], 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _slsqp_sparse(c0, r_init, time_budget=25.0, max_iter=300, ftol=1e-13):
    """SLSQP with only active (near-min) constraints via Delaunay + walls."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.03)

    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    # Precompute Delaunay edges for the initial configuration
    try:
        tri = Delaunay(c0)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    edges.add((min(i, j), max(i, j)))
    except Exception:
        edges = set()

    # Also add all pairs within 2.5 * r_init
    tree = cKDTree(c0)
    pairs_near = tree.query_pairs(2.5 * r_init, output_type='ndarray')
    for idx in pairs_near:
        i, j = int(idx[0]), int(idx[1])
        edges.add((min(i, j), max(i, j)))

    edge_list = sorted(edges)

    def _build_cons(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        # Find active pairs: distance < t + margin
        margin = max(0.005, 0.15 * t)
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        active_pairs = []
        for (i, j) in edge_list:
            if d[i, j] < 2 * (t + margin):
                active_pairs.append((i, j))
        # If no pairs found, use closest 20
        if not active_pairs:
            flat_d = d.ravel()
            idx_flat = np.argpartition(flat_d, 40)[:40]
            active_pairs = set()
            for fi in idx_flat:
                i, j = divmod(fi, N)
                if i != j:
                    active_pairs.add((min(i, j), max(i, j)))
            active_pairs = sorted(active_pairs)

        # Active walls
        wall_margin = max(0.005, 0.15 * t)
        active_walls = []
        for i in range(N):
            for axis in range(2):
                if 0.5 - pts[i, axis] < t + wall_margin:
                    active_walls.append((i, axis, 1))
                if pts[i, axis] + 0.5 < t + wall_margin:
                    active_walls.append((i, axis, -1))

        cons = []
        for i, j in active_pairs:
            def pc(x, i=i, j=j):
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                return np.linalg.norm(ci - cj) / 2.0 - x[2 * N]
            def pg(x, i=i, j=j):
                g = np.zeros(2 * N + 1)
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d > 1e-12:
                    g[2 * i:2 * i + 2] = diff / (2 * d)
                    g[2 * j:2 * j + 2] = -diff / (2 * d)
                g[2 * N] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': pc, 'jac': pg})

        for i, axis, sign in active_walls:
            if sign > 0:
                def wc(x, i=i, axis=axis):
                    return 0.5 - x[2 * i + axis] - x[2 * N]
                def wg(x, i=i, axis=axis):
                    g = np.zeros(2 * N + 1)
                    g[2 * i + axis] = -1.0
                    g[2 * N] = -1.0
                    return g
            else:
                def wc(x, i=i, axis=axis):
                    return x[2 * i + axis] + 0.5 - x[2 * N]
                def wg(x, i=i, axis=axis):
                    g = np.zeros(2 * N + 1)
                    g[2 * i + axis] = 1.0
                    g[2 * N] = -1.0
                    return g
            cons.append({'type': 'ineq', 'fun': wc, 'jac': wg})

        return cons

    x0 = np.concatenate([c0.ravel(), [r_init]])
    start = time.time()
    best_c = c0.copy()
    best_r = r_init

    # Multiple SLSQP passes with refreshed active set
    for pass_num in range(3):
        if time.time() - start > time_budget:
            break
        try:
            cons = _build_cons(x0)
            res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                           bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                           constraints=cons,
                           options={'maxiter': max_iter, 'ftol': ftol, 'disp': False})
            if res.x is not None and np.all(np.isfinite(res.x)):
                c_res = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
                s = _true_score(c_res)
                if s > best_r:
                    best_r, best_c = s, c_res
                    x0 = np.concatenate([best_c.ravel(), [best_r]])
                else:
                    # Try with the res.x even if score didn't improve (might be at boundary)
                    x0 = np.concatenate([np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5).ravel(), [max(res.x[2 * N], r_init)]])
        except Exception:
            break

    return best_c


def _active_set_newton(c0, r_target, time_budget=4.0, max_iter=200):
    """Active-set Newton: solve the KKT system for the active constraints."""
    c = np.array(c0, dtype=float, copy=True)
    r = r_target
    start = time.time()
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        # Compute distances
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        wall_d = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)  # (4, N)

        min_pair = d.min()
        min_wall = wall_d.min()
        tol = 1e-5

        # Active pairs
        active_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < min_pair + tol:
                    active_pairs.append((i, j))

        # Active walls
        active_walls = []
        for i in range(N):
            for axis in range(2):
                if wall_d[0, i] < min_wall + tol:  # 0.5 - c[i,0]
                    active_walls.append((i, 0, 1))
                if wall_d[1, i] < min_wall + tol:  # c[i,0] + 0.5
                    active_walls.append((i, 0, -1))
                if wall_d[2, i] < min_wall + tol:  # 0.5 - c[i,1]
                    active_walls.append((i, 1, 1))
                if wall_d[3, i] < min_wall + tol:  # c[i,1] + 0.5
                    active_walls.append((i, 1, -1))

        if not active_pairs and not active_walls:
            break

        # Build KKT: find direction that increases t while keeping active constraints tight
        # We solve: minimize ||delta||^2 subject to A @ delta = 0 (active constraints stay tight)
        # and b @ delta > 0 (t increases)
        # Simpler: just do a line search in the direction of the gradient of t

        # Direction: for each active pair, the constraint is |ci-cj|/2 >= t
        # For active wall: wall_dist >= t
        # We want to increase t, so we need to move points apart from active constraints

        n_vars = 2 * N
        # Build constraint Jacobian rows
        rows = []
        for i, j in active_pairs:
            diff = c[i] - c[j]
            dd = np.linalg.norm(diff)
            if dd < 1e-12:
                continue
            row = np.zeros(n_vars)
            row[2 * i:2 * i + 2] = diff / dd
            row[2 * j:2 * j + 2] = -diff / dd
            rows.append(row)
        for i, axis, sign in active_walls:
            row = np.zeros(n_vars)
            row[2 * i + axis] = sign
            rows.append(row)

        if not rows:
            break
        A = np.array(rows)
        m = A.shape[0]

        # We want a direction d such that A @ d = 0 (constraints remain tight)
        # and the t-component increases. Since t is coupled, we use:
        # The gradient of the objective (maximize t) projected onto null(A)
        # Simplified: use the null space of A
        # For small systems, just do a projected gradient step
        # Project e_t (the t-increase direction) onto null(A)
        # Actually, let's just do: find d in null(A) that maximizes t increase
        # t increase = sum of contributions... this is getting complex.
        # Simpler approach: just nudge along the average of active constraint normals

        # Average displacement direction
        disp = np.zeros(n_vars)
        for i, j in active_pairs:
            diff = c[i] - c[j]
            dd = np.linalg.norm(diff)
            if dd > 1e-12:
                disp[2 * i:2 * i + 2] += diff / dd
                disp[2 * j:2 * j + 2] -= diff / dd
        for i, axis, sign in active_walls:
            disp[2 * i + axis] += sign

        norm = np.linalg.norm(disp)
        if norm < 1e-12:
            break
        disp /= norm

        # Line search
        best_c = c.copy()
        best_r = _true_score(c)
        for step in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
            c_try = c + step * disp.reshape(N, 2)
            c_try = np.clip(c_try, -0.499, 0.499)
            s = _true_score(c_try)
            if s > best_r:
                best_r = s
                best_c = c_try.copy()
        c = best_c
        if best_r <= r + 1e-12:
            break

    return c


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B with soft-min objective."""
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

    k_schedule = [k, k * 2.0, k * 4.0]
    for kk in k_schedule:
        if time.time() - start_time > time_budget * 0.85:
            break
        try:
            res = minimize(lambda x: obj_and_grad(x, kk), c0.ravel(), jac=True,
                           method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 100, 'maxfun': 2000,
                                    'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0})
            c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
            s = _true_score(c)
            if s > best_r:
                best_r = s
                best_c = c.copy()
            c0 = c
        except Exception:
            break

    return best_c


def _multistart_force(c0, time_budget=5.0):
    """Binary search on radius with force relaxation."""
    r = _true_score(c0) * 0.9
    r = max(r, 0.05)
    lo, hi = 0.0, r
    cur = c0
    for _ in range(12):
        if time.time() - (time_budget - 5.0) > time_budget:  # rough check
            break
        cur = _force_relax(cur, r, time_budget=0.8, max_iter=3000)
        s = _true_score(cur)
        if s >= r - 1e-9:
            lo = r
            r = r * 1.05
            hi = r
        else:
            hi = r
            if hi - lo < 1e-7:
                break
            r = 0.5 * (lo + hi)
    return cur, lo


def construct():
    start_time = time.time()
    best = _init_grid()
    best_r = _true_score(best)

    # Multi-start force relaxation
    rng = np.random.default_rng(42)
    inits = [_init_grid(), _init_hex()]
    for _ in range(6):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    for c0 in inits:
        if time.time() - start_time > DEADLINE - 50:
            break
        cur, r = _multistart_force(c0, time_budget=3.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # Sparse SLSQP (Delaunay-based) - main optimizer
    if time.time() - start_time < DEADLINE - 25:
        budget = min(25.0, DEADLINE - 25 - (time.time() - start_time))
        if budget > 5.0:
            cur = _slsqp_sparse(best, best_r, time_budget=budget, max_iter=250, ftol=1e-13)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # Active-set Newton polish
    if time.time() - start_time < DEADLINE - 15:
        budget = min(4.0, DEADLINE - 15 - (time.time() - start_time))
        if budget > 1.0:
            cur = _active_set_newton(best, best_r, time_budget=budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # Soft-min L-BFGS polish
    if time.time() - start_time < DEADLINE - 10:
        budget = min(3.0, DEADLINE - 10 - (time.time() - start_time))
        if budget > 1.0:
            cur = _softmin_refine(best, k=100.0, time_budget=budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # Second sparse SLSQP with tighter tolerance
    if time.time() - start_time < DEADLINE - 8:
        budget = min(15.0, DEADLINE - 8 - (time.time() - start_time))
        if budget > 5.0:
            cur = _slsqp_sparse(best, best_r, time_budget=budget, max_iter=200, ftol=1e-15)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # Final active-set Newton
    if time.time() - start_time < DEADLINE - 4:
        budget = min(2.0, DEADLINE - 4 - (time.time() - start_time))
        if budget > 0.5:
            cur = _active_set_newton(best, best_r, time_budget=budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # Final soft-min with high k
    if time.time() - start_time < DEADLINE - 2:
        budget = min(1.5, DEADLINE - 2 - (time.time() - start_time))
        if budget > 0.5:
            cur = _softmin_refine(best, k=500.0, time_budget=budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # Final active-set
    if time.time() - start_time < DEADLINE - 0.5:
        budget = min(1.0, DEADLINE - 0.5 - (time.time() - start_time))
        if budget > 0.3:
            cur = _active_set_newton(best, best_r, time_budget=budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    return best
