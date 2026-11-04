"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])


def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)


def _neg_logsumexp_ratio(pts, temp):
    """
    Smooth surrogate to maximize: -soft_min(areas) / hull_area.
    soft_min(a) = -T * log(sum(exp(-a/T)))  -> approximates min(a) as T->0.
    We minimize this (i.e. maximize the negative).
    """
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    s = -a / temp
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * temp
    return soft_min / ha


def _neg_minmax_ratio(pts, lam=1e-3):
    """
    Min-max smoothing: penalize triangles that are near the minimum.
    Uses a soft-min with a moderate temperature on top of the exact min,
    weighted to emphasize the worst few triangles.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    # soft-min with small temperature to focus on the worst triangles
    T = 1e-4
    s = -a / T
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * T
    return soft_min / ha + lam * (np.mean(a) / ha)


def _halton(n, base):
    out = np.empty(n)
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        v = i
        while v > 0:
            v, d = divmod(v, base)
            x += d * f
            f /= base
        out[i - 1] = x
    return out


def _initial_halton(base1=2, base2=3):
    x = _halton(N, base1)
    y = _halton(N, base2)
    r = np.sqrt(x)
    th = 2 * np.pi * y
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _regular_polygon(n, r=1.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _refine(pts, time_budget, rng):
    """
    Refine a point set:
      1. L-BFGS on the smooth log-sum-exp objective (coarse -> fine temperature).
      2. Nelder-Mead polish on the exact min-max objective.
      3. Differential-evolution-style local search with random restarts.
    Returns best points and best score.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)

    # Stage 1: L-BFGS with decreasing temperature
    x0 = best.flatten()
    for temp in [0.03, 0.01, 0.003, 0.001]:
        if time.time() - t0 > time_budget - 4:
            break

        def obj(x):
            return _neg_logsumexp_ratio(x.reshape(N, 2), temp)

        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           options={'maxiter': 250, 'ftol': 1e-14,
                                    'gtol': 1e-9, 'maxls': 30})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
            x0 = cand
        except Exception:
            pass

    # Stage 2: Nelder-Mead on the min-max objective
    if time.time() - t0 < time_budget - 2:
        def obj2(x):
            return _neg_minmax_ratio(x.reshape(N, 2))
        try:
            res = minimize(obj2, best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-10,
                                    'fatol': 1e-12, 'initial_simplex': 1e-3})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    # Stage 3: DE-style local search
    if time.time() - t0 < time_budget - 1:
        pop = [best.copy()]
        for _ in range(5):
            pop.append(best + rng.normal(0, 0.05, (N, 2)))
        for it in range(40):
            if time.time() - t0 > time_budget - 0.5:
                break
            new_pop = []
            for i in range(len(pop)):
                if len(pop) >= 3:
                    a = pop[(i + 1) % len(pop)]
                    b = pop[(i + 2) % len(pop)]
                    c = pop[(i + 3) % len(pop)] if len(pop) > 3 else a
                    F = 0.8
                    trial = a + F * (b - c)
                else:
                    trial = pop[i] + rng.normal(0, 0.03, (N, 2))
                trial = trial + rng.normal(0, 0.01, (N, 2))
                s_t = _score(trial)
                s_i = _score(pop[i])
                new_pop.append(trial if s_t > s_i else pop[i])
            pop = new_pop
            # keep the best in pop
            best_in_pop = max(pop, key=lambda p: _score(p))
            s_bp = _score(best_in_pop)
            if s_bp > best_s:
                best_s = s_bp
                best = best_in_pop.copy()

    return best, best_s


def construct():
    """Main entry: multi-start refinement with DE-style search."""
    t0 = time.time()
    deadline = 145.0  # safety margin for 150s limit
    rng = np.random.RandomState(20240517)

    starts = [_regular_polygon(N, 1.0),
              _initial_halton(2, 3),
              _initial_halton(3, 5),
              _initial_halton(5, 7)]
    # jittered variants
    for off in [1, 2, 3]:
        h = _initial_halton(2, 3)
        starts.append(h + rng.normal(0, 0.1, (N, 2)))

    best_pts = None
    best_s = -1e9
    for st in starts:
        if time.time() - t0 > deadline - 10:
            break
        try:
            s = _score(st)
            if s > best_s:
                best_s = s
                best_pts = st.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _regular_polygon(N, 1.0)
        best_s = _score(best_pts)

    # Sort starts by initial score, best first
    scored = []
    for st in starts:
        try:
            scored.append((_score(st), st))
        except Exception:
            scored.append((0.0, st))
    scored.sort(key=lambda x: -x[0])

    # Allocate time: better starts get more
    n = len(scored)
    base = (deadline - 15.0) / n
    for i, (s0, st) in enumerate(scored):
        if time.time() - t0 > deadline - 5:
            break
        remaining = deadline - (time.time() - t0)
        quality = 1.0 + 0.5 * (s0 / max(best_s, 1e-9))
        alloc = min(base * quality, remaining - 2.0)
        if alloc < 3.0:
            continue
        try:
            op, os_ = _refine(st, alloc, rng)
            if os_ > best_s:
                best_s = os_
                best_pts = op.copy()
        except Exception:
            pass

    # Final refinement on the best
    if time.time() - t0 < deadline - 8:
        try:
            fp, fs = _refine(best_pts, deadline - (time.time() - t0) - 2, rng)
            if fs > best_s:
                best_s = fs
                best_pts = fp.copy()
        except Exception:
            pass

    return best_pts
