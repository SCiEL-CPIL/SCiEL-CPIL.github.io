import time
import numpy as np
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 15.0
_SQRT3 = np.sqrt(3.0)

# base flat-top unit hexagon vertices (CCW), circumradius 1
_BASE = np.array([
    [1.0, 0.0],
    [0.5, _SQRT3 / 2.0],
    [-0.5, _SQRT3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -_SQRT3 / 2.0],
    [0.5, -_SQRT3 / 2.0],
])


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _hex_vertices(x, y, ang_deg):
    a = np.deg2rad(ang_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return _BASE @ R.T + np.array([x, y])


def _all_vertices(inner):
    V = np.empty((N, 6, 2))
    for i in range(N):
        V[i] = _hex_vertices(inner[i][0], inner[i][1], inner[i][2])
    return V


def _outer_normals(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t + np.pi / 6 + k * np.pi / 3),
                      np.sin(t + np.pi / 6 + k * np.pi / 3)] for k in range(6)])


def _smooth_enclosing_side(V, C, theta_deg, tau=0.02):
    n = _outer_normals(theta_deg)            # (6,2)
    vn = np.einsum('ijk,kj->ik', V, n)       # (N,6)
    cn = n @ C                               # (6,)
    h = vn.max(axis=0) - cn                  # (6,)
    m = h.max()
    smax = m + tau * np.log(np.exp((h - m) / tau).sum())
    return (2.0 / _SQRT3) * smax


def _exact_enclosing_side(V, C, theta_deg):
    n = _outer_normals(theta_deg)
    vn = np.einsum('ijk,kj->ik', V, n)
    cn = n @ C
    h = vn.max(axis=0) - cn
    return (2.0 / _SQRT3) * h.max()


def _edge_normals(v):
    e = np.roll(v, -1, axis=0) - v
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return nrm


def _smooth_pair_overlap(V, i, j, tau=0.01):
    vi, vj = V[i], V[j]
    A = np.vstack([_edge_normals(vi), _edge_normals(vj)])   # (12,2)
    pi = vi @ A.T
    pj = vj @ A.T
    gap = np.maximum(pi.min(0), pj.min(0)) - np.minimum(pi.max(0), pj.max(0))
    m = (-gap).max()
    L = m + tau * np.log(np.exp((-gap - m) / tau).sum())    # ~ -separation
    return max(0.0, L)


def _smooth_total_overlap(V, tau=0.01):
    tot = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            tot += _smooth_pair_overlap(V, i, j, tau)
    return tot


def _exact_total_overlap(V):
    tot = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            vi, vj = V[i], V[j]
            A = np.vstack([_edge_normals(vi), _edge_normals(vj)])
            pi = vi @ A.T
            pj = vj @ A.T
            gap = np.maximum(pi.min(0), pj.min(0)) - np.minimum(pi.max(0), pj.max(0))
            sep = gap.max()
            if sep < 0.0:
                tot += -sep
    return tot


def _objective(params, theta_deg=0.0, lam=50.0):
    cx, cy = params[0], params[1]
    inner = [[params[2 + 3 * i], params[3 + 3 * i], params[4 + 3 * i]] for i in range(N)]
    V = _all_vertices(inner)
    S = _smooth_enclosing_side(V, np.array([cx, cy]), theta_deg)
    ov = _smooth_total_overlap(V)
    val = S + lam * ov * ov
    if not np.isfinite(val):
        return 1e6
    return val


def _initial_inner(perturb=0.0, rng=None):
    inner = []
    cols, rows = 3, 4
    for c in range(cols):
        x = 1.5 * c
        for r in range(rows):
            y = _SQRT3 * (r - (rows - 1) / 2.0)
            if c % 2 == 1:
                y += _SQRT3 / 2.0
            if perturb > 0 and rng is not None:
                x += rng.uniform(-perturb, perturb)
                y += rng.uniform(-perturb, perturb)
            inner.append([x, y, 0.0])
    return inner


def _params_from(inner, C):
    p = [C[0], C[1]]
    for (x, y, a) in inner:
        p += [x, y, a]
    return p


def _nelder(params0, lam, maxiter, theta=0.0):
    if _time_left() < 3.0:
        return params0
    res = minimize(_objective, params0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-5,
                            'adaptive': True, 'initial_simplex': None},
                   args=(theta, lam))
    return res.x if res.x is not None else params0


def _best_center(V, theta_deg, maxiter=80):
    n = _outer_normals(theta_deg)
    h0 = np.einsum('ijk,kj->ik', V, n).max(axis=0)   # (6,)
    def f(C):
        return (h0 - n @ C).max()
    C0 = np.array([V[:, :, 0].mean(), V[:, :, 1].mean()])
    res = minimize(f, C0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
    C = res.x
    S = (2.0 / _SQRT3) * (h0 - n @ C).max()
    return C, S


def construct():
    rng = np.random.default_rng(12345)

    # initial lattice + enclosing center
    inner0 = _initial_inner()
    V0 = _all_vertices(inner0)
    C0, S0 = _best_center(V0, 0.0)
    best_params = _params_from(inner0, C0)
    best_S = S0

    def evaluate(params):
        inner = [[params[2 + 3 * i], params[3 + 3 * i], params[4 + 3 * i]] for i in range(N)]
        V = _all_vertices(inner)
        C, S = _best_center(V, 0.0, maxiter=40)
        return S, V, C, inner

    S0, _, _, _ = evaluate(best_params)
    best_S = S0

    # ---- Phase A: multi-restart local search over (C, inner) ----
    n_restarts = 0
    while _time_left() > 25.0 and n_restarts < 14:
        n_restarts += 1
        if n_restarts == 1:
            p0 = best_params
        else:
            inner_p = _initial_inner(perturb=0.6, rng=rng)
            Cc = np.array([np.mean([q[0] for q in inner_p]),
                           np.mean([q[1] for q in inner_p])])
            p0 = _params_from(inner_p, Cc)
        p1 = _nelder(p0, lam=40.0, maxiter=140, theta=0.0)
        S1, _, _, _ = evaluate(p1)
        if S1 < best_S:
            best_S = S1
            best_params = p1

    # ---- Phase B: polish (crush overlap, tighten) ----
    if _time_left() > 12.0:
        p1 = _nelder(best_params, lam=200.0, maxiter=120, theta=0.0)
        S1, _, _, _ = evaluate(p1)
        if S1 < best_S + 1e-6:
            best_params = p1
            best_S = S1

    # ---- Phase C: separation safety (ensure disjoint) ----
    inner = [[best_params[2 + 3 * i], best_params[3 + 3 * i], best_params[4 + 3 * i]] for i in range(N)]
    V = _all_vertices(inner)
    if _exact_total_overlap(V) > 1e-6 and _time_left() > 10.0:
        def f_sep(params):
            inn = [[params[2 + 3 * i], params[3 + 3 * i], params[4 + 3 * i]] for i in range(N)]
            return _smooth_total_overlap(_all_vertices(inn), tau=0.005)
        res = minimize(f_sep, best_params, method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.x is not None:
            S1, _, _, _ = evaluate(res.x)
            if _exact_total_overlap(_all_vertices(
                    [[res.x[2 + 3 * i], res.x[3 + 3 * i], res.x[4 + 3 * i]] for i in range(N)])) <= 1e-5:
                best_params = res.x
                best_S = S1

    # ---- Phase D: exact outer refinement (scan rotation, solve center) ----
    inner = [[best_params[2 + 3 * i], best_params[3 + 3 * i], best_params[4 + 3 * i]] for i in range(N)]
    V = _all_vertices(inner)
    best_theta, best_C, best_side = 0.0, None, np.inf
    for th in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        C, S = _best_center(V, th, maxiter=120)
        if S < best_side:
            best_side, best_C, best_theta = S, C, th
    # refine around best_theta
    for th in [best_theta + d for d in (-4, -2, -1, 1, 2, 4)]:
        C, S = _best_center(V, th % 60.0, maxiter=120)
        if S < best_side:
            best_side, best_C, best_theta = S, C, th % 60.0

    # safety: tiny epsilon so containment holds under evaluator tolerance
    side_out = best_side * (1.0 + 1e-6) + 1e-6

    return inner, (float(best_C[0]), float(best_C[1])), float(side_out), float(best_theta)
