import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _angles_to_points(angles):
    """Convert angles (radians) to (n,2) point array on unit circle."""
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _normalize(points):
    """Subtract centroid, divide by mean distance to centroid. Return None if mean <= 0.01."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(points):
    """Check strictly convex: no repeated points, all turns nonzero same sign, total turn +-2*pi."""
    n = len(points)
    if n < 3:
        return False
    # No repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-12:
                return False
    # All cross products same sign and nonzero
    cross_products = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    if any(abs(c) < 1e-12 for c in cross_products):
        return False
    signs = [np.sign(c) for c in cross_products]
    if len(set(signs)) != 1:
        return False
    return True

def _vertex_score(v_idx, points):
    """Compute score for vertex v_idx."""
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    if sorted_dists[0] < 0.05:
        return 0.0
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = int(np.argmin(gaps))
    D = (sorted_dists[k] + sorted_dists[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k - 1:k + 3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _polygon_score(points):
    """Minimum vertex score over all vertices."""
    n = len(points)
    return min(_vertex_score(i, points) for i in range(n))

def _soft_min(scores, beta=30.0):
    """Soft minimum."""
    scores = np.asarray(scores, dtype=float)
    if np.any(scores < 0):
        return 0.0
    return -np.log(np.sum(np.exp(-beta * scores))) / beta

def _radial_to_points(coeffs):
    """Convert radial cosine coefficients to points.
    coeffs: array of length 10 (c_1..c_10) for r(theta) = 1 + sum c_k cos(k*theta)
    where theta is the base angle for each vertex."""
    n = 20
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    for k in range(1, 11):
        r += coeffs[k - 1] * np.cos(k * base_angles)
    # Ensure r > 0
    r = np.maximum(r, 0.5)
    return r * np.column_stack((np.cos(base_angles), np.sin(base_angles)))

def _neg_soft_score_from_coeffs(coeffs):
    """Negative soft-min score from radial coefficients."""
    points = _radial_to_points(coeffs)
    normalized = _normalize(points)
    if normalized is None:
        return 10.0
    if not _is_strictly_convex(normalized):
        return 10.0
    n = 20
    scores = np.array([_vertex_score(i, normalized) for i in range(n)])
    return -_soft_min(scores, beta=30.0)

def _neg_exact_score_from_coeffs(coeffs):
    """Negative exact min score from radial coefficients."""
    points = _radial_to_points(coeffs)
    normalized = _normalize(points)
    if normalized is None:
        return 10.0
    if not _is_strictly_convex(normalized):
        return 10.0
    n = 20
    scores = np.array([_vertex_score(i, normalized) for i in range(n)])
    return -np.min(scores)

def _regular_coeffs():
    """Zero coefficients = regular polygon."""
    return np.zeros(10)

def _jittered_coeffs(rng, scale):
    return rng.normal(0, scale, size=10)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 610  # safety margin before 630s

    n = 20
    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    def try_update(coeffs):
        nonlocal best_points, best_score
        points = _radial_to_points(coeffs)
        normalized = _normalize(points)
        if normalized is None:
            return
        if not _is_strictly_convex(normalized):
            return
        score = _polygon_score(normalized)
        if score > best_score:
            best_score = score
            best_points = normalized

    # Initialize with regular polygon
    try_update(_regular_coeffs())

    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 120:
        bounds = [(-0.3, 0.3)] * 10
        try:
            de_result = differential_evolution(
                _neg_soft_score_from_coeffs,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                workers=1,
                init='sobol',
                updating='deferred'
            )
            if de_result.x is not None:
                try_update(de_result.x)
        except Exception:
            pass

    # Phase 2: Local refinement from best and multiple random starts
    if best_points is not None and time.time() < deadline - 60:
        # Reconstruct approximate coeffs from best (just use best as reference)
        # Try L-BFGS-B from multiple starting points
        starts = [_regular_coeffs()]
        for scale in [0.2, 0.1, 0.05, 0.02, 0.01]:
            starts.append(_jittered_coeffs(rng, scale))

        for x0 in starts:
            if time.time() > deadline - 60:
                break
            try:
                res = minimize(
                    _neg_soft_score_from_coeffs,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000}
                )
                if res.x is not None:
                    try_update(res.x)
            except Exception:
                pass

    # Phase 3: Exact refinement
    if best_points is not None and time.time() < deadline - 30:
        # Use the best coeffs found - we need to track them
        # Re-run a small DE to find good coeffs, then refine
        try:
            bounds = [(-0.15, 0.15)] * 10
            de_result = differential_evolution(
                _neg_soft_score_from_coeffs,
                bounds,
                maxiter=100,
                popsize=10,
                tol=1e-8,
                seed=99,
                polish=True,
                workers=1,
                init='sobol'
            )
            if de_result.x is not None:
                try_update(de_result.x)
        except Exception:
            pass

    # Phase 4: Fine-tuning with exact objective
    if time.time() < deadline - 15:
        # Try small perturbations around best
        if best_points is not None:
            for _ in range(5):
                if time.time() > deadline - 15:
                    break
                x0 = _jittered_coeffs(rng, 0.01)
                try:
                    res = minimize(
                        _neg_exact_score_from_coeffs,
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 30000}
                    )
                    if res.x is not None:
                        try_update(res.x)
                except Exception:
                    pass

    # Fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = _normalize(_angles_to_points(angles))

    return best_points
