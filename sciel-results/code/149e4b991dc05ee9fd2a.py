# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _vertex_score(distances):
    """Compute the per-vertex score from sorted distances to other vertices."""
    d = np.sort(distances)
    # d has 19 elements
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k - 1, k, k + 1, k + 2]
    four_d = d[idx]
    raw = np.mean(np.maximum(four_d / D, D / four_d))
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon, min_score=0.0):
    """Compute the overall score: min vertex score over all vertices.
    If any vertex score < min_score, return early."""
    n = len(polygon)
    for i in range(n):
        v = polygon[i]
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        dists = np.delete(dists, i)
        s = _vertex_score(dists)
        if s < min_score:
            return s
    return min_score


def _is_valid_convex(polygon):
    """Check if polygon is strictly convex with no repeated points."""
    n = len(polygon)
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(polygon[i], polygon[j], atol=1e-10):
                return False
    signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    n = 20
    deadline = time.time() + 580

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _polygon_score(best_polygon)

    def _neg_softmin_score(params, beta=80.0):
        a = np.sort(params % (2 * np.pi))
        x = np.cos(a)
        y = np.sin(a)
        poly = np.stack([x, y], axis=1)
        if not _is_valid_convex(poly):
            return 1.0
        # Compute all vertex scores
        scores = []
        for i in range(n):
            v = poly[i]
            dists = np.sqrt(np.sum((poly - v) ** 2, axis=1))
            dists = np.delete(dists, i)
            scores.append(_vertex_score(dists))
        scores = np.array(scores)
        min_s = np.min(scores)
        shifted = scores - min_s
        soft_min = min_s - np.log(np.sum(np.exp(-beta * shifted))) / beta
        return -soft_min

    # Phase 1: Nelder-Mead with structured restarts
    rng = np.random.default_rng(42)
    x0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    restart_configs = [
        (x0.copy(), 3000, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.05, n), 2500, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.10, n), 2500, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.15, n), 2000, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.20, n), 2000, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.30, n), 1500, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.40, n), 1500, 1e-7, 1e-9),
        (x0 + rng.normal(0, 0.50, n), 1000, 1e-7, 1e-9),
    ]

    for x_start, maxiter, xatol, fatol in restart_configs:
        if time.time() > deadline - 120:
            break
        try:
            result = minimize(_neg_softmin_score, x_start, method='Nelder-Mead',
                            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True})
            if result.fun < 0:
                a = np.sort(result.x % (2 * np.pi))
                poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                if _is_valid_convex(poly):
                    exact_s = _polygon_score(poly, min_score=best_score)
                    if exact_s > best_score:
                        best_score = exact_s
                        best_polygon = poly
        except Exception:
            pass

    # Phase 2: COBYLA refinement
    if time.time() < deadline - 60:
        a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        a_best = np.sort(a_best % (2 * np.pi))

        try:
            result2 = minimize(_neg_softmin_score, a_best, method='COBYLA',
                             options={'maxiter': 8000, 'rhobeg': 1e-4, 'catol': 1e-10})
            if result2.fun < 0:
                a = np.sort(result2.x % (2 * np.pi))
                poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                if _is_valid_convex(poly):
                    exact_s = _polygon_score(poly, min_score=best_score)
                    if exact_s > best_score:
                        best_score = exact_s
                        best_polygon = poly
        except Exception:
            pass

    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 30:
        a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        a_best = np.sort(a_best % (2 * np.pi))
        a_cur = a_best.copy()
        cur_score = best_score

        for iteration in range(500):
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(n):
                # Try small perturbations of angle i
                for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
                    a_try = a_cur.copy()
                    a_try[i] += delta
                    a_try = np.sort(a_try % (2 * np.pi))
                    poly = np.stack([np.cos(a_try), np.sin(a_try)], axis=1)
                    if not _is_valid_convex(poly):
                        continue
                    s = _polygon_score(poly, min_score=cur_score)
                    if s > cur_score:
                        cur_score = s
                        a_cur = a_try
                        improved = True
                        break
            if not improved:
                break

        if cur_score > best_score:
            best_score = cur_score
            best_polygon = np.stack([np.cos(a_cur), np.sin(a_cur)], axis=1)

    return best_polygon


# EVOLVE-BLOCK-END
