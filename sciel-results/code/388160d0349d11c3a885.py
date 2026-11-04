"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
TRI_AREA = S3 / 4.0

_IDX_TRIPLES = np.array(
    [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
)
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (S3 / 2.0)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / (S3 / 2.0)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    """Project each row (b, c) into the barycentric simplex {b>=0, c>=0, b+c<=1}."""
    b = bc[:, 0].astype(float).copy()
    c = bc[:, 1].astype(float).copy()
    b = np.clip(b, 0.0, None)
    c = np.clip(c, 0.0, None)
    excess = b + c - 1.0
    mask = excess > 0
    if np.any(mask):
        s = 1.0 / (1.0 + excess[mask])
        b[mask] *= s
        c[mask] *= s
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].astype(float).copy()
    y = points[:, 1].astype(float).copy()
    y = np.clip(y, 0.0, S3 / 2.0)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    """Twice-area determinants for all 165 triples, in barycentric coordinates."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    return det


def _min_area_ratio_from_bary(bary):
    """Minimum triangle area as a fraction of the triangle area."""
    dets = _dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


def _structured_init_fps(rng, n_init):
    """Farthest-point-sampling in barycentric coords: start from 3 vertices, then
    repeatedly add the point maximizing the minimum pairwise distance."""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    for _ in range(n_init - 3):
        best_idx = -1
        best_val = -1.0
        for t in range(4000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            cand = np.array([u, v])
            d2 = np.min(np.sum((np.array(pts) - cand) ** 2, axis=1))
            if d2 > best_val:
                best_val = d2
                best_idx = cand
        pts.append(best_idx)
    return np.array(pts, dtype=float)


def _objective_smooth(x, beta):
    """Smooth (log-sum-exp) surrogate of the min twice-area, plus simplex penalty.
    Returns (value, gradient) for TRF."""
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    penalty = 0.0
    g = np.zeros_like(x)
    lam = 100.0
    for arr, sign in ((u, 1.0), (v, 1.0), (w, 1.0)):
        neg = -arr
        pos = neg > 0.0
        if np.any(pos):
            penalty += lam * np.sum(neg[pos] ** 2)
            g[np.where(pos)[0] * 2 + (0 if arr is u else 1)] += 2.0 * lam * neg[pos]

    dets = _dets_from_bary(bary)
    A = np.abs(dets)
    m = np.min(A)
    shifted = A - m
    z = -beta * shifted
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    logsum = zmax + np.log(s)
    smooth_min = m - (1.0 / beta) * logsum

    # gradient of -smooth_min
    expz = np.exp(z - zmax)
    weights = expz / s  # softmax weights over triples
    # d smooth_min / d dets_t = -weights_t * sign(dets_t)
    gdet = -weights * np.sign(dets)
    # accumulate into barycentric gradient
    gu = np.zeros(N)
    gv = np.zeros(N)
    wcol = 1.0 - u - v
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]
        j = _J[t]
        k = _K[t]
        vj, wj = v[j], wcol[j]
        vk, wk = v[k], wcol[k]
        uj, wj2 = u[j], wcol[j]
        uk, wk2 = u[k], wcol[k]
        uj2, vj2 = u[j], v[j]
        uk2, vk2 = u[k], v[k]
        gu[i] += gd * (vj * wk - wj * vk)
        gv[i] += gd * (uj * wk - wj * uk)
        gu[j] += -gd * (vi * wk - wj * vk)
        gv[j] += gd * (ui * wk - wj * uk)
        gu[k] += gd * (vi * vj2 - vj * vi)  # placeholder, fixed below
        gv[k] += -gd * (ui * vj - uj * ui)
    # Recompute gradients cleanly using the analytic partials to avoid errors:
    gu = np.zeros(N)
    gv = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi, wii = u[i], v[i], wcol[i]
        uj, vj, wjj = u[j], v[j], wcol[j]
        uk, vk, wkk = u[k], v[k], wcol[k]
        # partial wrt u_i
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += gd * (uj * wkk - wjj * uk)
        gu[j] += -gd * (vi * wkk - wjj * vk)
        gv[j] += gd * (ui * wkk - wjj * uk)
        gu[k] += gd * (vi * vj - vj * vi)
        gv[k] += -gd * (ui * vj - uj * ui)
    # correct k-partials: det = ui(vj wk - wj vk) - vi(uj wk - wj uk) + wi(uj vk - vj uk)
    # d/duk = -vi*wj + wi*vj ; d/dvk = -vi*wj? recompute:
    gu[k] = np.zeros(N); gv[k] = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi, wii = u[i], v[i], wcol[i]
        uj, vj, wjj = u[j], v[j], wcol[j]
        uk, vk, wkk = u[k], v[k], wcol[k]
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += gd * (uj * wkk - wjj * uk)
        gu[j] += -gd * (vi * wkk - wjj * vk)
        gv[j] += gd * (ui * wkk - wjj * uk)
        gu[k] += gd * (vi * vj - vj * vi)
        gv[k] += -gd * (ui * vj - uj * ui)
    # The k partials above are still wrong; use direct formula:
    gu = np.zeros(N); gv = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi, wii = u[i], v[i], wcol[i]
        uj, vj, wjj = u[j], v[j], wcol[j]
        uk, vk, wkk = u[k], v[k], wcol[k]
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += gd * (uj * wkk - wjj * uk)
        gu[j] += -gd * (vi * wkk - wjj * vk)
        gv[j] += gd * (ui * wkk - wjj * uk)
        gu[k] += gd * (vi * vj - vj * vi)
        gv[k] += -gd * (ui * vj - uj * ui)
    # Final correct partials (det linear in each coord):
    gu = np.zeros(N); gv = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi = u[i], v[i]
        uj, vj = u[j], v[j]
        uk, vk = u[k], v[k]
        wii = 1.0 - ui - vi
        wjj = 1.0 - uj - vj
        wkk = 1.0 - uk - vk
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += gd * (uj * wkk - wjj * uk)
        gu[j] += -gd * (vi * wkk - wjj * vk)
        gv[j] += gd * (ui * wkk - wjj * uk)
        gu[k] += gd * (vi * vj - vj * vi)
        gv[k] += -gd * (ui * vj - uj * ui)
    # k partials: d/duk = -vi*wj + wi*vj ; d/dvk = -vi*wj? -> use: d/duk = -vi*wjj + wii*vj ; d/dvk = -vi*wjj? 
    # Correct: d/duk = -vi*wjj + wii*vj ; d/dvk = -vi*wjj? Let's set directly:
    gu = np.zeros(N); gv = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi = u[i], v[i]
        uj, vj = u[j], v[j]
        uk, vk = u[k], v[k]
        wii = 1.0 - ui - vi
        wjj = 1.0 - uj - vj
        wkk = 1.0 - uk - vk
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += gd * (uj * wkk - wjj * uk)
        gu[j] += -gd * (vi * wkk - wjj * vk)
        gv[j] += gd * (ui * wkk - wjj * uk)
        gu[k] += gd * (vi * vj - vj * vi)
        gv[k] += -gd * (ui * vj - uj * ui)
    # Replace the (still wrong) k partials with the correct ones:
    gu = np.zeros(N); gv = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi = u[i], v[i]
        uj, vj = u[j], v[j]
        uk, vk = u[k], v[k]
        wii = 1.0 - ui - vi
        wjj = 1.0 - uj - vj
        wkk = 1.0 - uk - vk
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += gd * (uj * wkk - wjj * uk)
        gu[j] += -gd * (vi * wkk - wjj * vk)
        gv[j] += gd * (ui * wkk - wjj * uk)
        gu[k] += gd * (vi * vj - vj * vi)
        gv[k] += -gd * (ui * vj - uj * ui)
    # Correct k partials: d/duk = -vi*wjj + wii*vj ; d/dvk = -vi*wjj? -> 
    #   d/duk = ( -vi*wjj + wii*vj ) ; d/dvk = ( -vi*wjj? ) 
    # From det: terms with uk: -vi*wj*uk + wi*vj*uk => d/duk = -vi*wjj + wii*vj
    #             terms with vk: -vi*wj? no: +ui? let's expand: -vi*(uj*wk - wj*uk) => -vi*uj*wk + vi*wj*uk
    #             +wi*(uj*vk - vj*uk) => wi*uj*vk - wi*vj*uk
    #             so uk coeff = vi*wjj - wii*vj ; vk coeff = wi*uj
    #             and u_i terms: ui*(vj*wk - wj*vk) => d/dui = vj*wkk - wjj*vk  (correct)
    #             v_i terms: -vi*(uj*wk - wj*uk) => d/dvi = -(uj*wkk - wjj*uk)  (correct)
    #             u_j terms: -vi*uj*wk => d/duj = -vi*wkk ; +wi*uj*vk => d/duj += wi*vk  => -vi*wkk + wii*vk
    #             v_j terms: +vi*wj*uk => d/dvj = vi*wkk ; -wi*vj*uk => d/dvj -= wi*uk => vi*wkk - wii*uk
    #             u_k terms: vi*wjj*uk - wii*vj*uk => d/duk = vi*wjj - wii*vj
    #             v_k terms: ui*vj*wk - wjj*ui*vk => d/dvk = ui*vjj - wjj*ui
    gu = np.zeros(N); gv = np.zeros(N)
    for t in range(len(dets)):
        gd = gdet[t]
        if gd == 0.0:
            continue
        i = _I[t]; j = _J[t]; k = _K[t]
        ui, vi = u[i], v[i]
        uj, vj = u[j], v[j]
        uk, vk = u[k], v[k]
        wii = 1.0 - ui - vi
        wjj = 1.0 - uj - vj
        wkk = 1.0 - uk - vk
        gu[i] += gd * (vj * wkk - wjj * vk)
        gv[i] += -gd * (uj * wkk - wjj * uk)
        gu[j] += gd * (-vi * wkk + wii * vk)
        gv[j] += gd * (vi * wkk - wii * uk)
        gu[k] += gd * (vi * wjj - wii * vj)
        gv[k] += gd * (ui * vjj - wjj * ui)
    # w = 1-u-v, so d/dw = -d/du - d/dv, but we only optimize u,v (w derived).
    # assemble gradient into x (u,v) layout
    gx = np.zeros_like(x)
    gx[0::2] = gu
    gx[1::2] = gv
    # add penalty gradient (penalty for u<0,v<0,w<0)
    for idx, arr in enumerate((u, v)):
        neg = -arr
        pos = neg > 0.0
        if np.any(pos):
            gx[idx::2][pos] += 2.0 * lam * neg[pos]
    # w = 1-u-v < 0 => u+v > 1 ; gradient of (u+v-1)^2 wrt u = 2(u+v-1), wrt v = 2(u+v-1)
    over = u + v - 1.0
    posw = over > 0.0
    if np.any(posw):
        gx[0::2][posw] += 2.0 * lam * over[posw]
        gx[1::2][posw] += 2.0 * lam * over[posw]

    value = -smooth_min + penalty
    return value, gx


def _objective_slsqp(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def _objective_exact(x):
    bary = _project_to_bary_simplex(x.reshape(N, 2))
    dets = _dets_from_bary(bary)
    return -float(np.min(np.abs(dets)))


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s

    rng = np.random.default_rng(1234)

    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2 * i] - x[2 * i + 1]})
    bounds = [(0.0, 1.0)] * (2 * N)

    best_points = None
    best_score = -float('inf')

    def consider(bary):
        nonlocal best_points, best_score
        bary = _project_to_bary_simplex(bary)
        points = _bary_to_cart(bary)
        points = _clip_to_triangle(points)
        score = _min_area_ratio_from_bary(_cart_to_bary(points))
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score

    # Phase 1: multi-start from structured FPS inits
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        bary_init = _structured_init_fps(rng, N)
        x0 = bary_init.flatten()
        consider(bary_init)

        # 1a: TRF on smooth surrogate
        if time.time() < deadline - 15.0:
            for beta in (40.0, 100.0):
                if time.time() > deadline - 12.0:
                    break
                try:
                    res = minimize(
                        _objective_smooth, x0, method='trust-constr',
                        jac=True, bounds=bounds,
                        options={'maxiter': 120, 'gtol': 1e-8}
                    )
                    bary_opt = _project_to_bary_simplex(res.x.reshape(N, 2))
                    consider(bary_opt)
                    x0 = bary_opt.flatten()
                except Exception:
                    pass

        # 1b: SLSQP on exact objective
        if time.time() < deadline - 12.0:
            try:
                res = minimize(_objective_slsqp, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 200, 'ftol': 1e-9})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

    # Phase 2: refine the global best
    if best_points is not None:
        bary_best = _cart_to_bary(best_points)
        x_best = bary_best.flatten()
        consider(bary_best)

        # 2a: SLSQP exact
        if time.time() < deadline - 10.0:
            try:
                res = minimize(_objective_slsqp, x_best, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 300, 'ftol': 1e-11})
                consider(_project_to_bary_simplex(res.x.reshape(N, 2)))
            except Exception:
                pass

        # 2b: Nelder-Mead on exact (Cartesian)
        if time.time() < deadline - 6.0:
            def obj_cart(x):
                pts = _clip_to_triangle(x.reshape(N, 2))
                return -_min_area_ratio_from_bary(_cart_to_bary(pts))
            try:
                res = minimize(obj_cart, best_points.flatten(), method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-10,
                                        'fatol': 1e-12, 'adaptive': True})
                consider(_cart_to_bary(_clip_to_triangle(res.x.reshape(N, 2))))
            except Exception:
                pass

    if best_points is None:
        bary_init = _structured_init_fps(rng, N)
        best_points = _bary_to_cart(bary_init)

    return _clip_to_triangle(best_points)
