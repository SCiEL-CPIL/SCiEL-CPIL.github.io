import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2
DEADLINE = 140.0  # safety margin before the 150 s kill


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _pairwise_dists_sq(P):
    """Vector of squared pairwise distances (upper triangle)."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, 1)
    return d_sq[iu]


def _score(P):
    """Exact objective: (max_dist / min_dist)^2 = max_d2 / min_d2."""
    d_sq = _pairwise_dists_sq(P)
    min_d = np.min(d_sq)
    max_d = np.max(d_sq)
    if min_d < 1e-12:
        return 1e10
    return max_d / min_d


# ---------------------------------------------------------------------------
# Smooth surrogate (soft max / soft min of squared distances)
# ---------------------------------------------------------------------------
def _soft_objective(params, T=0.01):
    """Smooth surrogate. Point 0 is fixed at origin; params hold the other 15."""
    P = np.zeros((N, 2))
    P[1:] = params.reshape(-1, 2)
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)

    m = np.max(d_sq)
    soft_max = m + T * np.log(np.sum(np.exp((d_sq - m) / T)))
    m2 = np.min(d_sq)
    soft_min = m2 - T * np.log(np.sum(np.exp(-(d_sq - m2) / T)))
    if soft_min < 1e-10:
        return 1e10
    return soft_max / soft_min


def _soft_gradient(params, T=0.01):
    """Analytical gradient of the soft objective w.r.t. the free points (1..15)."""
    P = np.zeros((N, 2))
    P[1:] = params.reshape(-1, 2)

    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq_full = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq = d_sq_full[iu]
    d_sq = np.maximum(d_sq, 1e-12)

    # soft-max weights (softmax over d_sq / T)
    ds = d_sq / T
    mm = np.max(ds)
    e = np.exp(ds - mm)
    sw = e / np.sum(e)
    soft_max = T * (mm + np.log(np.sum(e)))

    # soft-min weights (softmax over -d_sq / T)
    nds = -d_sq / T
    mn = np.max(nds)
    en = np.exp(nds - mn)
    mw = en / np.sum(en)
    soft_min = -T * (mn + np.log(np.sum(en)))

    if soft_min < 1e-10:
        return np.zeros_like(params)

    # d(soft_max/soft_min)/d d_sq
    g_d = (sw * soft_min + soft_max * mw) / (T * soft_min * soft_min)

    grad_P = np.zeros((N, 2))
    g0, g1 = iu[0], iu[1]
    for idx in range(len(g0)):
        i, j = g0[idx], g1[idx]
        g = g_d[idx]
        grad_P[i] += 2.0 * g * (P[i] - P[j])
        grad_P[j] += 2.0 * g * (P[j] - P[i])

    return grad_P[1:].flatten()


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _hex_points():
    """All hexagonal-lattice points within radius 2 (the 19-point 3-ring cluster)."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p * p) <= 4.0 + 1e-9:
                pts.append(p)
    return np.array(pts)


def _hex_subsets():
    """All 16-point subsets of the 19-point hex cluster (19 combinations)."""
    hexp = _hex_points()
    n = len(hexp)
    out = []
    for omit in range(n):
        keep = [k for k in range(n) if k != omit]
        out.append(hexp[keep])
    return out


def _get_initial_candidates():
    cands = []

    # 1. Hexagonal lattice subsets (the strong structural family)
    cands.extend(_hex_subsets())

    # 2. 4x4 square grid
    gx = np.array([0.0, 1.0, 2.0, 3.0])
    gy = np.array([0.0, 1.0, 2.0, 3.0])
    X, Y = np.meshgrid(gx, gy)
    cands.append(np.column_stack([X.flatten(), Y.flatten()]))

    # 3. Two concentric rings (8 inner, 8 outer), a few radius choices
    for ri, ro in [(0.5, 1.5), (0.6, 1.4), (0.7, 1.3), (1.0, 2.0)]:
        pts = []
        for k in range(8):
            a = 2 * np.pi * k / 8
            pts.append([ro * np.cos(a), ro * np.sin(a)])
        for k in range(8):
            a = 2 * np.pi * (k + 0.5) / 8
            pts.append([ri * np.cos(a), ri * np.sin(a)])
        cands.append(np.array(pts))

    # 4. Fibonacci (golden-angle) spiral
    ga = np.pi * (3 - np.sqrt(5))
    sp = []
    for k in range(16):
        r = np.sqrt(k + 1) * 0.6
        a = k * ga
        sp.append([r * np.cos(a), r * np.sin(a)])
    cands.append(np.array(sp))

    # 5. Random perturbations of the grid
    rng = np.random.default_rng(42)
    base = cands[19]  # the 4x4 grid
    for _ in range(6):
        cands.append(base + rng.normal(0, 0.15, base.shape))

    # 6. A few random Gaussian clusters
    for s in [7, 13, 21]:
        rt = np.random.default_rng(s)
        cands.append(rt.normal(0, 1, (N, 2)))

    return cands


def _prepare(P):
    """Translate so point 0 is at the origin; return free params (points 1..15)."""
    P = P - P[0]
    return P[1:].flatten()


def _reconstruct(params):
    P = np.zeros((N, 2))
    P[1:] = params.reshape(-1, 2)
    return P


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()

    candidates = _get_initial_candidates()

    best_params = None
    best_score = np.inf
    records = []  # (score, params)

    # ---- Phase 1: annealed soft refinement from many diverse starts ----------
    temperatures = [0.2, 0.05, 0.01, 0.003]

    for i, P0 in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 30:
            break
        try:
            cur = _prepare(P0)
            cur_score = _score(_reconstruct(cur))
            if cur_score < best_score:
                best_score = cur_score
                best_params = cur.copy()

            for T in temperatures:
                if time.time() - t0 > DEADLINE - 30:
                    break
                res = minimize(
                    lambda x: _soft_objective(x, T),
                    lambda x: _soft_gradient(x, T),
                    cur,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-9},
                )
                sc = _score(_reconstruct(res.x))
                if sc < cur_score:
                    cur_score = sc
                    cur = res.x.copy()
                if sc < best_score:
                    best_score = sc
                    best_params = res.x.copy()
        except Exception:
            continue

        records.append((cur_score, cur.copy()))

    # ---- Phase 2: sharp refinement of the best several starts ---------------
    records.sort(key=lambda r: r[0])
    for sc, params in records[:6]:
        if time.time() - t0 > DEADLINE - 12:
            break
        try:
            res = minimize(
                lambda x: _soft_objective(x, 0.001),
                lambda x: _soft_gradient(x, 0.001),
                params,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-11},
            )
            sc2 = _score(_reconstruct(res.x))
            if sc2 < best_score:
                best_score = sc2
                best_params = res.x.copy()
        except Exception:
            pass

    # ---- Phase 3: exact local polish (nonsmooth) with tight tolerances ------
    if best_params is not None and time.time() - t0 < DEADLINE - 4:
        try:
            res = minimize(
                lambda x: _score(_reconstruct(x)),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True},
            )
            sc = _score(_reconstruct(res.x))
            if sc < best_score:
                best_score = sc
                best_params = res.x.copy()
        except Exception:
            pass

    # ---- Assemble the final answer ------------------------------------------
    if best_params is None:
        best_params = _prepare(candidates[0])

    final_pts = _reconstruct(best_params)

    # Validity guard: fall back to the best raw candidate if something broke.
    if _score(final_pts) > 1e6:
        fb = None
        fb_sc = np.inf
        for P0 in candidates:
            s = _score(P0)
            if s < fb_sc:
                fb_sc = s
                fb = P0
        final_pts = fb if fb is not None else candidates[0]

    return final_pts
