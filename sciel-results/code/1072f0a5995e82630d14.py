import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 142.0  # explicit wall-clock safety margin before the 150 s kill

# Precompute all 286 triples once
_ALL_TRIPLES = np.array(list(combinations(range(N), 3)))


def _all_triangle_areas(points):
    """Vectorized areas of every triangle (shape (286,))."""
    p1 = points[_ALL_TRIPLES[:, 0]]
    p2 = points[_ALL_TRIPLES[:, 1]]
    p3 = points[_ALL_TRIPLES[:, 2]]
    cross = ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
             - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.abs(cross)


def _min_triangle_area(points):
    if len(points) < 3:
        return 0.0
    return float(np.min(_all_triangle_areas(points)))


def _hull_area(points):
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 1e-9


def _score(points):
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_a


# ---------------- structured radial-jitter parameterization ----------------
# points = center + radius_i * (cos theta_i, sin theta_i)
# theta_i = 2*pi*i/N + dtheta_i  (uniform angular spacing + jitter)
# This is a constrained relaxation of the regular N-gon: it keeps a structured,
# high-symmetry topology while breaking exact symmetry so gradient methods work.

def _reconstruct(x):
    cx = x[0]
    cy = x[1]
    radii = x[2:2 + N]
    dtheta = x[2 + N:2 + 2 * N]
    base = 2.0 * np.pi * np.arange(N) / N
    theta = base + dtheta
    return np.column_stack([cx + radii * np.cos(theta),
                            cy + radii * np.sin(theta)])


def _soft_min_triangle_area(points, beta):
    areas = _all_triangle_areas(points)
    areas = np.clip(areas, 1e-15, None)
    z = -beta * areas
    z = z - np.max(z)
    log_sum = np.log(np.sum(np.exp(z)))
    return float(-(np.max(-beta * areas) + log_sum) / beta)


def _convexity_barrier(points):
    """Penalize (small) triangles that violate strict convexity of the hull order."""
    hull_a = _hull_area(points)
    areas = _all_triangle_areas(points)
    # triangles far smaller than the hull suggest near-collinearity / concavity
    frac = np.clip(areas / (hull_a + 1e-12), 0.0, None)
    # count how many are below a small threshold (concave-ish)
    return float(np.sum(frac < 1e-4))


def _soft_objective(x, beta):
    pts = _reconstruct(x)
    hull_a = _hull_area(pts)
    if hull_a < 1e-8:
        return 1e10
    sm = _soft_min_triangle_area(pts, beta)
    if sm < 1e-15:
        return 1e10
    val = -sm / hull_a
    # light convexity barrier
    val += 1e-3 * _convexity_barrier(pts)
    return val


def _exact_objective(x):
    pts = _reconstruct(x)
    hull_a = _hull_area(pts)
    if hull_a < 1e-8:
        return 1e10
    return -_min_triangle_area(pts) / hull_a + 1e-3 * _convexity_barrier(pts)


def _regular_n_gon():
    """Regular 13-gon encoded in the structured parameter space."""
    x = np.zeros(2 + 2 * N)
    x[2:2 + N] = 1.0
    return x


def _random_structured(rng, radius_scale=0.15, theta_scale=0.12):
    x = np.zeros(2 + 2 * N)
    x[2:2 + N] = 1.0 + rng.normal(0, radius_scale, N)
    x[2 + N:2 + 2 * N] = rng.normal(0, theta_scale, N)
    return x


def _powell_refine(f, x0, deadline, maxiter=200):
    """Coordinate-descent (Powell) refinement on the structured parameter space."""
    x = x0.copy()
    best = f(x)
    n = len(x)
    for _ in range(maxiter):
        if time.time() >= deadline:
            break
        improved = False
        for k in range(n):
            if time.time() >= deadline:
                break
            step = 0.05
            for _ in range(30):
                if time.time() >= deadline:
                    break
                cand = x.copy()
                cand[k] += step
                fc = f(cand)
                if fc < best:
                    x = cand
                    best = fc
                    improved = True
                    step *= 1.5
                else:
                    cand = x.copy()
                    cand[k] -= step
                    fc = f(cand)
                    if fc < best:
                        x = cand
                        best = fc
                        improved = True
                        step *= 1.5
                    else:
                        step *= 0.5
        if not improved:
            break
    return x, best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def _remaining():
        return deadline - time.time()

    rng = np.random.default_rng(7)

    # ---- diverse structured starts ----
    starts = [_regular_n_gon()]
    # offset centers
    for dx in [-0.08, 0.08, -0.04, 0.04]:
        for dy in [-0.08, 0.08, -0.04, 0.04]:
            x = _regular_n_gon().copy()
            x[0] = dx
            x[1] = dy
            starts.append(x)
    # random structured
    for rs, ts in [(0.05, 0.05), (0.10, 0.08), (0.15, 0.12), (0.20, 0.15)]:
        for _ in range(6):
            starts.append(_random_structured(rng, rs, ts))

    best_x = None
    best_score = -np.inf

    # evaluate starts (exact)
    for x in starts:
        if _remaining() < 5.0:
            break
        pts = _reconstruct(x)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_x = x.copy()

    # ---- Phase 1: L-BFGS-B with soft-min, progressive beta ----
    for beta in [40.0, 80.0, 160.0, 320.0]:
        if _remaining() < 18.0:
            break
        # rank starts by exact score
        ranked = sorted(starts, key=lambda x: _score(_reconstruct(x)), reverse=True)
        for x0 in ranked[:6]:
            if _remaining() < 6.0:
                break
            try:
                res = minimize(_soft_objective, x0, args=(beta,),
                               method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                s = _score(_reconstruct(res.x))
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
            except Exception:
                pass

    # ---- Phase 2: Nelder-Mead polish (exact) from best ----
    if _remaining() > 15.0:
        try:
            res = minimize(_exact_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9,
                                    'fatol': 1e-11, 'adaptive': True})
            s = _score(_reconstruct(res.x))
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass

    # ---- Phase 3: coordinate-descent (Powell) refinement ----
    if _remaining() > 12.0:
        x_new, _ = _powell_refine(_exact_objective, best_x, deadline, maxiter=150)
        s = _score(_reconstruct(x_new))
        if s > best_score:
            best_score = s
            best_x = x_new.copy()

    # ---- Phase 4: a couple of Nelder-Mead restarts from good structured starts ----
    if _remaining() > 10.0:
        ranked = sorted(starts, key=lambda x: _score(_reconstruct(x)), reverse=True)
        for x0 in ranked[:3]:
            if _remaining() < 6.0:
                break
            x0 = x0 + rng.normal(0, 0.03, len(x0))
            try:
                res = minimize(_exact_objective, x0, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-9,
                                        'fatol': 1e-11, 'adaptive': True})
                s = _score(_reconstruct(res.x))
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
            except Exception:
                pass

    # ---- Phase 5: final high-beta L-BFGS-B + coordinate-descent polish ----
    if _remaining() > 6.0:
        try:
            res = minimize(_soft_objective, best_x, args=(600.0,),
                           method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            s = _score(_reconstruct(res.x))
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    if _remaining() > 4.0:
        x_new, _ = _powell_refine(_exact_objective, best_x, deadline, maxiter=80)
        s = _score(_reconstruct(x_new))
        if s > best_score:
            best_score = s
            best_x = x_new.copy()

    pts = _reconstruct(best_x)
    # safety: ensure a valid 13x2 array
    if pts.shape != (N, 2):
        pts = _regular_n_gon() and _reconstruct(_regular_n_gon())
    return pts
