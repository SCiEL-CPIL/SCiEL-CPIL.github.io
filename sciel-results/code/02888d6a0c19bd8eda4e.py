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


def _polygon_score(polygon):
    """Compute the overall score: min vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        v = polygon[i]
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        # Exclude self-distance
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return min(scores)


def _is_valid_convex(polygon):
    """Check if polygon is strictly convex with no repeated points."""
    n = len(polygon)
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(polygon[i], polygon[j], atol=1e-10):
                return False
    # Check all turns have same sign and are nonzero
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
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Start from regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _polygon_score(best_polygon)
    
    def _score_from_params(params):
        """Convert parameters to polygon and compute score."""
        a = np.sort(params % (2 * np.pi))
        x = np.cos(a)
        y = np.sin(a)
        poly = np.stack([x, y], axis=1)
        if not _is_valid_convex(poly):
            return 0.0
        return _polygon_score(poly)
    
    def _softmin_score(polygon, beta=50.0):
        """
        Compute a smooth surrogate for min vertex score using log-sum-exp.
        This allows gradient-based optimization.
        """
        n = len(polygon)
        scores = []
        for i in range(n):
            v = polygon[i]
            dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
            dists = np.delete(dists, i)
            scores.append(_vertex_score(dists))
        scores = np.array(scores)
        # Soft-min: -log(sum(exp(-beta * s)))/beta approximates min(s)
        # For numerical stability
        min_s = np.min(scores)
        shifted = scores - min_s
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min = min_s - np.log(np.sum(np.exp(-beta * shifted))) / beta
        return soft_min
    
    def _neg_softmin_score(params):
        a = np.sort(params % (2 * np.pi))
        x = np.cos(a)
        y = np.sin(a)
        poly = np.stack([x, y], axis=1)
        if not _is_valid_convex(poly):
            return 1.0  # Large penalty
        return -_softmin_score(poly, beta=50.0)
    
    # Phase 1: Nelder-Mead with multiple restarts (broad exploration)
    rng = np.random.default_rng(42)
    x0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 120:
            break
        if r == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.normal(0, 0.08, n)
        
        try:
            result = minimize(_neg_softmin_score, x_start, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if result.fun < 0:
                s = -result.fun
                if s > best_score:
                    a = np.sort(result.x % (2 * np.pi))
                    poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                    if _is_valid_convex(poly):
                        exact_s = _polygon_score(poly)
                        if exact_s > best_score:
                            best_score = exact_s
                            best_polygon = poly
        except Exception:
            pass
    
    # Phase 2: Targeted refinement on the best polygon found
    if time.time() < deadline - 60:
        # Use the best polygon as starting point for fine-tuning
        a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        a_best = np.sort(a_best % (2 * np.pi))
        
        # Try COBYLA for more precise local optimization
        try:
            result2 = minimize(_neg_softmin_score, a_best, method='COBYLA',
                             options={'maxiter': 5000, 'rhobeg': 1e-4, 'catol': 1e-10})
            if result2.fun < 0:
                s = -result2.fun
                a = np.sort(result2.x % (2 * np.pi))
                poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                if _is_valid_convex(poly):
                    exact_s = _polygon_score(poly)
                    if exact_s > best_score:
                        best_score = exact_s
                        best_polygon = poly
        except Exception:
            pass
        
        # Phase 3: Final high-precision polish with very tight tolerances
        if time.time() < deadline - 30:
            a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
            a_best = np.sort(a_best % (2 * np.pi))
            
            try:
                result3 = minimize(_neg_softmin_score, a_best, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
                if result3.fun < 0:
                    s = -result3.fun
                    a = np.sort(result3.x % (2 * np.pi))
                    poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                    if _is_valid_convex(poly):
                        exact_s = _polygon_score(poly)
                        if exact_s > best_score:
                            best_score = exact_s
                            best_polygon = poly
            except Exception:
                pass
    
    return best_polygon


# EVOLVE-BLOCK-END
