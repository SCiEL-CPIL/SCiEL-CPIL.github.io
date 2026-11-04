"""Combined packing: KDTree force relaxation + Delaunay SLSQP epigraph + adaptive soft-min."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay

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


def _force_relax_kdtree(c, r, time_budget=3.0, max_iter=20000):
    """KDTree-accelerated force relaxation for hard-disk packing."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
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


def _softmin_refine(c0, k, time_budget=3.0, maxiter=200, maxfun=5000,
                    ftol=1e-15, gtol=1e-10, adaptive_k=True):
    """Smooth L-BFGS-B polish with adaptive k schedule."""
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

    if adaptive_k:
        k_schedule = [k, k * 1.5, k * 2.0]
    else:
        k_schedule = [k]

    for kk in k_schedule:
        if time.time() - start_time > time_budget * 0.9:
            break
        try:
            res = minimize(lambda x: obj_and_grad(x, kk), c0.ravel(), jac=True,
                           method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': maxiter, 'maxfun': maxfun,
                                    'ftol': ftol, 'gtol': gtol,
                                    'iprint': 0})
            c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
            s = _true_score(c)
            if s > best_r:
                best_r = s
                best_c = c.copy()
            c0 = c
        except Exception:
            break

    return best_c


def _delaunay_contact_graph(c, r):
    """Return set of (i,j) pairs that are near-contact in the Delaunay graph."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    edges.add((min(i, j), max(i, j)))
        near = set()
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2.5 * r:
                near.add((i, j))
        return near
    except Exception:
        tree = cKDTree(c)
        pairs = tree.query_pairs(2.8 * r, output_type='ndarray')
        return set((min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs)


def _slsqp_epigraph(c0, r0, time_budget=5.0, maxiter=200, ftol=1e-14):
    """SLSQP epigraph: maximize t s.t. t <= d_ij/2, t <= wall distances."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r0, 0.01)

    contacts = _delaunay_contact_graph(c0, r0)
    contact_list = list(contacts)
    n_contacts = len(contact_list)

    def obj(x):
        return -x[-1]

    def obj_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g

    def constraint_fun(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[-1]
        pair_cons = np.zeros(n_contacts)
        for idx, (i, j) in enumerate(contact_list):
            diff = pts[i] - pts[j]
            d = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            pair_cons[idx] = d / 2.0 - t
        wall_cons = np.zeros(4 * N)
        wall_cons[0::4] = pts[:, 0] + 0.5 - t
        wall_cons[1::4] = 0.5 - pts[:, 0] - t
        wall_cons[2::4] = pts[:, 1] + 0.5 - t
        wall_cons[3::4] = 0.5 - pts[:, 1] - t
        return np.concatenate([pair_cons, wall_cons])

    def constraint_jac(x):
        pts = x[:2 * N].reshape(N, 2)
        n_total = n_contacts + 4 * N
        J = np.zeros((n_total, 2 * N + 1))
        for idx, (i, j) in enumerate(contact_list):
            diff = pts[i] - pts[j]
            d = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            if d < 1e-12:
                continue
            u0 = diff[0] / d
            u1 = diff[1] / d
            J[idx, 2 * i] = 0.5 * u0
            J[idx, 2 * i + 1] = 0.5 * u1
            J[idx, 2 * j] = -0.5 * u0
            J[idx, 2 * j + 1] = -0.5 * u1
            J[idx, -1] = -1.0
        base = n_contacts
        for i in range(N):
            J[base + 4 * i, 2 * i] = 1.0
            J[base + 4 * i, -1] = -1.0
            J[base + 4 * i + 1, 2 * i] = -1.0
            J[base + 4 * i + 1, -1] = -1.0
            J[base + 4 * i + 2, 2 * i + 1] = 1.0
            J[base + 4 * i + 2, -1] = -1.0
            J[base + 4 * i + 3, 2 * i + 1] = -1.0
            J[base + 4 * i + 3, -1] = -1.0
        return J

    x0 = np.concatenate([c0.ravel(), [r0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    cons = {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}

    try:
        start = time.time()
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'iprint': 0})
        if time.time() - start > time_budget:
            return c0
        c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
        return c
    except Exception:
        return c0


def _langevin_dynamics(c0, r, time_budget=6.0, max_iter=50000, rng=None):
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
    best_c = c.copy()
    best_score = _true_score(c)
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        T = T_init * (T_min / T_init) ** (i / max(1, max_iter))
        disp = np.zeros_like(c)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        if len(pairs) > 0:
            for idx in pairs:
                ii, jj = int(idx[0]), int(idx[1])
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


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) KDTree force relaxation with binary search on radius
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 35:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 15:
                break
            cur = _force_relax_kdtree(cur, r, time_budget=2.0, max_iter=8000)
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

    # 2) Langevin dynamics for diversification
    if time.time() - start_time < DEADLINE - 30:
        c_lang = _langevin_dynamics(best, best_r, time_budget=8.0, max_iter=30000, rng=rng)
        s = _true_score(c_lang)
        if s > best_r:
            best_r, best = s, c_lang

    # 3) Adaptive soft-min polish from multiple candidates
    candidates = [best]
    candidates.append(_force_relax_kdtree(best, best_r, time_budget=1.5, max_iter=5000))
    for c0 in inits[:3]:
        candidates.append(_force_relax_kdtree(c0, best_r, time_budget=1.5, max_iter=5000))

    for k in (80.0, 150.0, 250.0, 400.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 35:
                break
            c = _softmin_refine(c0, k, time_budget=3.0, adaptive_k=True)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 4) Primary SLSQP epigraph with generous budget
    if time.time() - start_time < DEADLINE - 30:
        c = _slsqp_epigraph(best, best_r, time_budget=20.0, maxiter=500, ftol=1e-15)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 5) Soft-min intermediate polish
    if time.time() - start_time < DEADLINE - 20:
        c = _softmin_refine(best, 200.0, time_budget=5.0,
                            maxiter=300, maxfun=8000,
                            ftol=1e-16, gtol=1e-11, adaptive_k=True)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 6) Second SLSQP with tighter tolerance
    if time.time() - start_time < DEADLINE - 12:
        c = _slsqp_epigraph(best, best_r, time_budget=8.0, maxiter=400, ftol=1e-16)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 7) Final high-precision SLSQP
    if time.time() - start_time < DEADLINE - 6:
        c = _slsqp_epigraph(best, best_r, time_budget=4.0, maxiter=600, ftol=1e-17)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 8) Ultra-precise final soft-min polish
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 500.0, time_budget=2.0,
                            maxiter=200, maxfun=6000,
                            ftol=1e-18, gtol=1e-14, adaptive_k=True)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
