"""Packing 200 equal circles in a unit square — refined soft-min + epigraph polish."""
N = 200
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_vals(vals, T):
    """Numerically stable log-sum-exp soft minimum."""
    m = np.min(vals)
    if vals.size == 1:
        return m
    return m - T * np.log(np.sum(np.exp(-(vals - m) / T)))


def _energy(params, T):
    """Smooth surrogate: maximize min(pair/2, boundary) via soft-min."""
    c = np.clip(params.reshape(-1, 2), -0.499999, 0.499999)
    T = max(T, 1e-8)

    b = 0.5 - np.abs(c)
    b_min_all = np.min(b)
    b_sub = b[b < b_min_all + 0.08]
    r_bound = _soft_min_vals(b_sub, T)

    d = pdist(c, metric='euclidean')
    d_min_all = np.min(d)
    d_sub = d[d < d_min_all + 0.08]
    r_pair = _soft_min_vals(d_sub, T) / 2.0

    r = _soft_min_vals(np.array([r_pair, r_bound]), T)
    return -r


def _init_hex():
    """Best hexagonal packing within a range of spacings."""
    best_c, best_r = None, 0.0
    for d in np.linspace(0.054, 0.076, 30):
        pts = []
        rws = d * np.sqrt(3) / 2.0
        off = d / 2.0
        y = -0.5 + d / 2.0
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2.0
            if row % 2 == 1:
                x += off
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append((x, y))
                x += d
            y += rws
            row += 1
        if len(pts) >= N:
            c = np.array(pts[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r, best_c = r, c.copy()
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        t = (np.arange(m) + 0.5) / m - 0.5
        g = np.array([(x, y) for y in t for x in t])
        best_c, best_r = g[:N].copy(), _compute_radius(g[:N])
    return best_c


def _init_jittered_grid(seed):
    """Jittered square-grid initialisation."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    gx, gy = np.meshgrid(np.linspace(-0.44, 0.44, m), np.linspace(-0.44, 0.44, m))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    rng.shuffle(g)
    c = g[:N].copy() + rng.normal(0, 0.012, size=(N, 2))
    return np.clip(c, -0.499999, 0.499999)


def _lbfgs_pass(c0, T, max_time):
    """One L-BFGS-B pass on the soft-min surrogate."""
    def obj(x):
        return _energy(x, T)
    try:
        res = minimize(obj, c0.flatten(), method='L-BFGS-B',
                       bounds=[(-0.499999, 0.499999)] * (2 * N),
                       options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14,
                                'maxls': 200, 'maxfun': 40000})
        return np.clip(res.x.reshape(-1, 2), -0.499999, 0.499999)
    except Exception:
        return c0.copy()


def _local_opt(c0, max_time):
    """Annealed L-BFGS-B schedule."""
    t0 = time.time()
    best_c, best_r = c0.copy(), _compute_radius(c0)
    cur = c0.copy()
    for T in (0.04, 0.025, 0.015, 0.01, 0.007, 0.005, 0.004, 0.003, 0.002, 0.0015):
        if time.time() - t0 > max_time - 1.0:
            break
        step_time = min(6.0, max(2.0, max_time - (time.time() - t0) - 1.0))
        c_new = _lbfgs_pass(cur, T, step_time)
        c_new = np.clip(c_new, -0.499999, 0.499999)
        cur = c_new
        r = _compute_radius(c_new)
        if r > best_r:
            best_r, best_c = r, c_new.copy()
    return best_c, best_r


def _polish(c0, max_time):
    """Very-low-temperature final polish."""
    t0 = time.time()
    best_c, best_r = c0.copy(), _compute_radius(c0)
    cur = c0.copy()
    for T in (0.001, 0.0006, 0.0004, 0.00025, 0.00015, 0.0001, 0.00006, 0.00004, 0.000025, 0.000015):
        if time.time() - t0 > max_time - 0.5:
            break
        step_time = min(3.5, max(1.5, max_time - (time.time() - t0) - 0.5))
        c_new = _lbfgs_pass(cur, T, step_time)
        cur = np.clip(c_new, -0.499999, 0.499999)
        r = _compute_radius(cur)
        if r > best_r:
            best_r, best_c = r, cur.copy()
    return best_c, best_r


def _pdist_to_ij(k, n):
    i = int((n - 1 - np.sqrt(8.0 * k + 1.0)) // 2)
    j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
    return i, j


def _slsqp(c0, max_time):
    """Epigraph SLSQP: variable t, hard distance & boundary constraints on the active set."""
    t0 = time.time()
    best_c, best_r = c0.copy(), _compute_radius(c0)
    cur_c, cur_r = c0.copy(), best_r

    def _obj(x):
        return -x[0]

    def _grad(x):
        g = np.zeros_like(x)
        g[0] = -1.0
        return g

    def _build_cons(c_ref, t_ref):
        cons = []
        d = pdist(c_ref, metric='euclidean')
        close = np.where(d < 2.4 * t_ref)[0]
        for k in close:
            i, j = _pdist_to_ij(k, N)
            def pf(x, _i=i, _j=j):
                ci = x[1 + 2*_i:1 + 2*_i + 2]
                cj = x[1 + 2*_j:1 + 2*_j + 2]
                return np.linalg.norm(ci - cj) - 2.0 * x[0]
            def pj(x, _i=i, _j=j):
                ci = x[1 + 2*_i:1 + 2*_i + 2]
                cj = x[1 + 2*_j:1 + 2*_j + 2]
                diff = ci - cj
                dd = max(np.linalg.norm(diff), 1e-12)
                g = np.zeros_like(x)
                g[0] = -2.0
                g[1 + 2*_i:1 + 2*_i + 2] = diff / dd
                g[1 + 2*_j:1 + 2*_j + 2] = -diff / dd
                return g
            cons.append({'type': 'ineq', 'fun': pf, 'jac': pj})

        for i in range(N):
            for k in range(2):
                if 0.5 - abs(c_ref[i, k]) < 1.6 * t_ref:
                    def bf(x, _i=i, _k=k):
                        return 0.5 - abs(x[1 + 2*_i + _k]) - x[0]
                    def bj(x, _i=i, _k=k):
                        g = np.zeros_like(x)
                        g[0] = -1.0
                        v = x[1 + 2*_i + _k]
                        g[1 + 2*_i + _k] = -np.sign(v) if v != 0 else 1.0
                        return g
                    cons.append({'type': 'ineq', 'fun': bf, 'jac': bj})
        return cons

    for _ in range(4):
        if time.time() - t0 > max_time - 2.0:
            break
        try:
            cons = _build_cons(cur_c, cur_r)
            x0 = np.concatenate([[cur_r], cur_c.flatten()])
            res = minimize(_obj, x0, jac=_grad, method='SLSQP',
                           bounds=[(cur_r * 0.95, 0.4999)] + [(-0.499999, 0.499999)] * (2 * N),
                           constraints=cons,
                           options={'maxiter': 250, 'ftol': 1e-13})
            r_new = res.x[0]
            c_new = np.clip(res.x[1:].reshape(-1, 2), -0.499999, 0.499999)
            r_check = _compute_radius(c_new)
            if r_check >= r_new * 0.9995 and r_check > best_r:
                best_r, best_c = r_check, c_new.copy()
                cur_c, cur_r = c_new, r_check
        except Exception:
            continue
    return best_c, best_r


def construct():
    t_start = time.time()
    deadline = t_start + 145.0   # 5s safety margin under 150s limit

    def remaining():
        return deadline - time.time()

    # --- Initialization ---
    c_hex = _init_hex()
    r_hex = _compute_radius(c_hex)
    best_c, best_r = c_hex.copy(), r_hex

    # --- Primary local optimization ---
    if remaining() > 45:
        c1, r1 = _local_opt(best_c, max_time=min(35.0, remaining() - 20.0))
        if r1 > best_r:
            best_r, best_c = r1, c1.copy()

    # --- Restart 1: jittered grid ---
    if remaining() > 35:
        c2 = _init_jittered_grid(7)
        r2 = _compute_radius(c2)
        if r2 > best_r:
            best_r, best_c = r2, c2.copy()
        if remaining() > 30:
            c2b, r2b = _local_opt(c2, max_time=min(20.0, remaining() - 15.0))
            if r2b > best_r:
                best_r, best_c = r2b, c2b.copy()

    # --- Restart 2: different jitter ---
    if remaining() > 25:
        c3 = _init_jittered_grid(123)
        r3 = _compute_radius(c3)
        if r3 > best_r:
            best_r, best_c = r3, c3.copy()
        if remaining() > 20:
            c3b, r3b = _local_opt(c3, max_time=min(15.0, remaining() - 10.0))
            if r3b > best_r:
                best_r, best_c = r3b, c3b.copy()

    # --- SLSQP epigraph polish ---
    if remaining() > 18:
        cs, rs = _slsqp(best_c, max_time=min(15.0, remaining() - 8.0))
        if rs > best_r:
            best_r, best_c = rs, cs.copy()

    # --- Final low-T polish ---
    if remaining() > 6:
        cp, rp = _polish(best_c, max_time=min(8.0, remaining() - 2.0))
        if rp > best_r:
            best_r, best_c = rp, cp.copy()

    # --- One more SLSQP pass if time allows ---
    if remaining() > 10:
        cs2, rs2 = _slsqp(best_c, max_time=min(8.0, remaining() - 3.0))
        if rs2 > best_r:
            best_r, best_c = rs2, cs2.copy()

    # --- Final clip & validity check ---
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _init_hex()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    return best_c
