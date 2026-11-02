"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 120.0
_T0 = None


def _hull_area(pts):
    try:
        h = ConvexHull(pts)
        return float(h.volume)
    except Exception:
        return 0.0


def _min_tri_area(pts):
    n = len(pts)
    # Vectorized cross products: area = 0.5*|cross|
    # Use all triples
    best = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if a < best:
                    best = a
    return best


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 0:
        return 0.0
    return _min_tri_area(pts) / ha


def _softmin_surrogate(pts, beta):
    """Return -softmin of triangle areas (to maximize)."""
    n = len(pts)
    areas = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(a)
    areas = np.array(areas)
    # softmin via log-sum-exp: softmin = -logsumexp(-beta*areas)/beta
    m = np.min(areas)
    s = np.sum(np.exp(-beta * (areas - m)))
    softmin = m - np.log(s) / beta
    return -softmin


def _neg_softmin(x, beta):
    pts = x.reshape(N, 2)
    ha = _hull_area(pts)
    if ha <= 0:
        return 1e6
    sm = _softmin_surrogate(pts, beta)
    # maximize min area / hull area -> minimize -sm/ha
    return -sm / ha


def _unit_circle_init(seed):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.2, 0.2, N)
    r = 1.0 + rng.uniform(-0.05, 0.05, N)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _local_polish(pts, time_left):
    """Refine with Powell on the softmin surrogate then exact."""
    global _T0
    best = pts.copy()
    best_score = _score(pts)
    betas = [5.0, 10.0, 20.0, 40.0]
    for beta in betas:
        if time.time() - _T0 > _DEADLINE - 10:
            break
        x0 = best.ravel()
        try:
            res = minimize(_neg_softmin, x0, args=(beta,), method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best = cand
                best_score = s
        except Exception:
            pass
    return best, best_score


def construct():
    global _T0
    _T0 = time.time()

    # Phase 1: global differential evolution on softmin surrogate
    best = _unit_circle_init(42)
    best_score = _score(best)

    if time.time() - _T0 > _DEADLINE - 60:
        return best

    # DE over [0,1]^26
    try:
        bounds = [(0.0, 1.0)] * (2 * N)
        de_res = differential_evolution(
            _neg_softmin, bounds, args=(10.0,),
            seed=7, maxiter=60, tol=1e-6,
            popsize=12, mutation=(0.3, 1.0), recombination=0.9,
            polish=False, init='latinhypercube',
            updating='deferred', workers=1
        )
        if de_res.x is not None:
            cand = de_res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best = cand
                best_score = s
    except Exception:
        pass

    # Phase 2: local polish
    if time.time() - _T0 < _DEADLINE - 15:
        best, best_score = _local_polish(best, _DEADLINE - time.time())

    # Final safety: ensure valid
    if _hull_area(best) <= 0:
        return _unit_circle_init(0)
    return best
# EVOLVE-BLOCK-END
