# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all triples (C(11,3) = 165)
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N)
                         for j in range(i + 1, N)
                         for k in range(j + 1, N)])
N_TRIPLES = len(_IDX_TRIPLES)
_I_IDX = _IDX_TRIPLES[:, 0]
_J_IDX = _IDX_TRIPLES[:, 1]
_K_IDX = _IDX_TRIPLES[:, 2]

# ---------------------------------------------------------------------------
# Coordinate / geometry helpers
# ---------------------------------------------------------------------------

def _bary_to_cart(u, v):
    """Convert barycentric (u, v) -> Cartesian. w = 1-u-v. P = u*V0 + v*V1 + w*V2."""
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))


def _all_dets_from_bary(bary):
    """All 165 determinants (twice the area in barycentric coords)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    Pi = P[_I_IDX]
    Pj = P[_J_IDX]
    Pk = P[_K_IDX]
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det


def _clamp_bary(bary):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary[:, 0])
    v = np.maximum(0.0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))


def _min_area_ratio(bary):
    """Minimum |det| over all triples (proportional to area / TRI_AREA)."""
    dets = _all_dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def _hard_objective(x):
    """Negative min |det| (nonsmooth)."""
    bary = _clamp_bary(x.reshape(N, 2))
    return -_min_area_ratio(bary)


def _softmin_objective(x, beta=120.0):
    """Smooth surrogate: -soft-min(|det|) + floor penalty (for L-BFGS-B)."""
    bary = _clamp_bary(x.reshape(N, 2))
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - log_sum / beta
    floor_penalty = 1000.0 * np.sum(np.maximum(0.0, 1e-6 - areas) ** 2)
    return -soft_min + floor_penalty


# ---------------------------------------------------------------------------
# Epigraph formulation:  maximize t  s.t.  t <= |det| for all triples
#   x layout: [u0,v0,...,u10,v10, t]  (2N + 1 variables)
#   objective: -t  (smooth!)
#   constraints: 1 - u_i - v_i >= 0  and  |det| - t >= 0  (smooth)
# ---------------------------------------------------------------------------

def _epigraph_obj(x):
    return -x[2 * N]


def _epigraph_jac_obj(x):
    g = np.zeros(2 * N + 1)
    g[2 * N] = -1.0
    return g


def _epigraph_constraints():
    cons = []
    # Simplex constraints: 1 - u_i - v_i >= 0
    for i in range(N):
        ui, vi = 2 * i, 2 * i + 1
        cons.append({
            'type': 'ineq',
            'fun': lambda x, ui=ui, vi=vi: 1.0 - x[ui] - x[vi],
            'jac': lambda x, ui=ui, vi=vi: np.concatenate(
                [np.zeros(ui), np.array([-1.0, -1.0]), np.zeros(2 * N + 1 - ui - 2)]),
        })
    # Area constraints: |det| - t >= 0
    for idx in range(N_TRIPLES):
        a, b, c = int(_IDX_TRIPLES[idx, 0]), int(_IDX_TRIPLES[idx, 1]), int(_IDX_TRIPLES[idx, 2])
        ia, ib, ic = 2 * a, 2 * b, 2 * c
        cons.append({
            'type': 'ineq',
            'fun': lambda x, ia=ia, ib=ib, ic=ic, idx=idx:
                np.abs(_all_dets_from_bary(x[:2 * N].reshape(N, 2))[idx]) - x[2 * N],
            'jac': lambda x, ia=ia, ib=ib, ic=ic, idx=idx:
                _epigraph_area_jac(x, idx, ia, ib, ic),
        })
    return cons


def _epigraph_area_jac(x, idx, ia, ib, ic):
    """Gradient of |det_idx| - t w.r.t. x.
    d(det)/du_a = v_b*w_c - v_c*w_b ; d(det)/du_b = w_a*v_c - w_c*v_a ; d(det)/du_c = w_a*v_b - w_b*v_a
    (and cyclic for v). det in barycentric = 2*area_bary (matches _all_dets)."""
    u = x[:N]
    v = x[N:2 * N]
    w = 1.0 - u - v
    a, b, c = ia // 2, ib // 2, ic // 2
    det = _all_dets_from_bary(x[:2 * N].reshape(N, 2))[idx]
    if det < 0:
        s = -1.0
    else:
        s = 1.0
    g = np.zeros(2 * N + 1)
    # d(det)/du_a
    g[ia] = s * (v[b] * w[c] - v[c] * w[b])
    # d(det)/du_b
    g[ib] = s * (w[a] * v[c] - w[c] * v[a])
    # d(det)/du_c
    g[ic] = s * (w[a] * v[b] - w[b] * v[a])
    # d(det)/dv_a
    g[N + ia] = s * (w[b] * v[c] - w[c] * v[b])
    # d(det)/dv_b
    g[N + ib] = s * (w[a] * w[c] - w[c] * w[a])
    # d(det)/dv_c
    g[N + ic] = s * (w[a] * w[b] - w[b] * w[a])
    # d(-t)
    g[2 * N] = -1.0
    return g


_EPI_CONS = _epigraph_constraints()


def _polish_epigraph(bary0, maxiter=400, ftol=1e-12):
    """SLSQP on the smooth epigraph problem. Returns (bary, t)."""
    t0 = max(_min_area_ratio(bary0) * 0.99, 1e-8)
    x0 = np.concatenate([bary0.flatten(), [t0]])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 1.0)]
    try:
        res = minimize(
            _epigraph_obj, x0, method='SLSQP',
            bounds=bounds, constraints=_EPI_CONS, jac=_epigraph_jac_obj,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False},
        )
        bary = _clamp_bary(res.x[:2 * N].reshape(N, 2))
        t = float(res.x[2 * N])
        return bary, t
    except Exception:
        return bary0.copy(), t0


def _polish_slsqp_hard(bary0, maxiter=200, ftol=1e-12):
    """SLSQP on the hard (nonsmooth) objective with simplex constraints."""
    cons = []
    for i in range(N):
        ui, vi = 2 * i, 2 * i + 1
        cons.append({
            'type': 'ineq',
            'fun': lambda x, ui=ui, vi=vi: 1.0 - x[ui] - x[vi],
            'jac': lambda x, ui=ui, vi=vi: np.concatenate(
                [np.zeros(ui), np.array([-1.0, -1.0]), np.zeros(2 * N - ui - 2)]),
        })
    x0 = bary0.flatten()
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _hard_objective, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False},
        )
        return _clamp_bary(res.x.reshape(N, 2))
    except Exception:
        return bary0.copy()


def _polish_nm(bary0, maxiter=400):
    """Nelder-Mead on the hard objective (robust, no constraints needed)."""
    x0 = bary0.flatten()
    try:
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10,
                     'adaptive': True, 'disp': False},
        )
        return _clamp_bary(res.x.reshape(N, 2))
    except Exception:
        return bary0.copy()


def _polish_lbfgs_softmin(bary0, beta=120.0, maxiter=400):
    """L-BFGS-B on the smooth soft-min surrogate."""
    x0 = bary0.flatten()
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _softmin_objective, x0, args=(beta,), method='L-BFGS-B',
            bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15,
                                    'gtol': 1e-10, 'maxls': 50, 'disp': False},
        )
        return _clamp_bary(res.x.reshape(N, 2))
    except Exception:
        return bary0.copy()


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _fps_init():
    """Farthest Point Sampling on a barycentric grid."""
    n_grid = 22
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            candidates.append([i / n_grid, j / n_grid])
    candidates = np.array(candidates)
    selected = [np.array([1 / 3, 1 / 3])]
    for _ in range(N - 1):
        S = np.array(selected)
        d = np.linalg.norm(candidates[:, None, :] - S[None, :, :], axis=2)
        d = d.min(axis=1)
        selected.append(candidates[int(np.argmax(d))])
    return np.array(selected)


def _random_init(seed):
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                pts[i] = [u, v]
                break
    return pts


def _structured_inits():
    inits = [_fps_init()]
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1 / 3, 1 / 3], [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]])
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33], [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]])
    pD = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.333, 0.0], [0.667, 0.0], [0.0, 0.333], [0.0, 0.667],
        [0.333, 0.333], [0.667, 0.333], [0.333, 0.667], [0.5, 0.5]])
    pE = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1 / 3, 1 / 3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25]])
    pF = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3],
        [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75]])
    for p in (pA, pB, pD, pE, pF):
        inits.append(_clamp_bary(p.copy()))
    rng = np.random.RandomState(7)
    inits.append(_clamp_bary(pD + rng.normal(0, 0.02, pD.shape)))
    rng2 = np.random.RandomState(11)
    inits.append(_clamp_bary(pE + rng2.normal(0, 0.02, pE.shape)))
    for s in (42, 123, 456):
        inits.append(_random_init(s))
    return inits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def tleft():
        return deadline - time.time()

    best_bary = _fps_init()
    best_score = _min_area_ratio(best_bary)

    def consider(bary):
        nonlocal best_bary, best_score
        sc = _min_area_ratio(bary)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
        return sc

    inits = _structured_inits()

    # Phase 1: multi-method local refinement from structured inits
    for init in inits:
        if tleft() < 12:
            break
        cur = _clamp_bary(init.copy())
        consider(cur)
        # 1) epigraph (smooth) SLSQP
        if tleft() > 12:
            b, _ = _polish_epigraph(cur, maxiter=300, ftol=1e-11)
            consider(b)
            cur = b
        # 2) SLSQP hard
        if tleft() > 12:
            consider(_polish_slsqp_hard(cur, maxiter=200, ftol=1e-12))
        # 3) Nelder-Mead
        if tleft() > 12:
            consider(_polish_nm(cur, maxiter=400))
        # 4) L-BFGS-B soft-min (anneal beta)
        if tleft() > 12:
            c2 = cur.copy()
            for beta in (60.0, 150.0, 400.0):
                if tleft() < 8:
                    break
                c2 = _polish_lbfgs_softmin(c2, beta=beta, maxiter=300)
                consider(c2)

    # Phase 2: global Differential Evolution
    if tleft() > 40:
        def de_obj(x):
            bary = _clamp_bary(x.reshape(N, 2))
            s = _min_area_ratio(bary)
            return -s if s >= 0 else 1.0
        try:
            res = differential_evolution(
                de_obj, [(0.0, 1.0)] * (2 * N),
                seed=42, maxiter=40, popsize=18, tol=1e-6,
                mutation=(0.4, 1.0), recombination=0.8,
                polish=False, init='sobol', updating='deferred',
                disp=False, workers=1,
            )
            de_bary = _clamp_bary(res.x.reshape(N, 2))
            consider(de_bary)
            # polish the DE result with the smooth epigraph
            b, _ = _polish_epigraph(de_bary, maxiter=400, ftol=1e-12)
            consider(b)
        except Exception:
            pass

    # Phase 3: a few random restarts if time remains
    if tleft() > 25:
        for s in (2024, 999, 31337, 555, 8080):
            if tleft() < 15:
                break
            r = _random_init(s)
            consider(r)
            b, _ = _polish_epigraph(r, maxiter=200, ftol=1e-11)
            consider(b)
            consider(_polish_nm(b, maxiter=300))

    # Phase 4: final high-precision polish on the best solution
    if tleft() > 20:
        budget = min(45.0, tleft() - 10)
        cur = best_bary.copy()
        for _ in range(3):
            if tleft() < 12:
                break
            b, _ = _polish_epigraph(cur, maxiter=600, ftol=1e-14)
            consider(b)
            if np.allclose(b, cur, atol=1e-14):
                break
            cur = b
        if tleft() > 12:
            consider(_polish_slsqp_hard(cur, maxiter=400, ftol=1e-14))
        if tleft() > 12:
            consider(_polish_nm(cur, maxiter=1500))

    cart = _bary_to_cart(best_bary[:, 0], best_bary[:, 1])
    return cart
# EVOLVE-BLOCK-END
