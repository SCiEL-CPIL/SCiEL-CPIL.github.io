# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _vertex_score_raw(polygon):
    """Compute the raw per-vertex score (before min(1, .)) for each vertex.
    Returns an array of length n with the raw score (can be 0).
    """
    n = polygon.shape[0]
    # pairwise distances
    diff = polygon[:, None, :] - polygon[None, :, :]
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    scores = np.zeros(n)
    for i in range(n):
        # distances to other vertices, sorted
        d = np.sort(dist[i].copy())
        d = d[1:]  # exclude self (d_0..d_18)
        if np.any(d < 0.05):
            scores[i] = 0.0
            continue
        # gaps
        gaps = d[1:] - d[:-1]  # d_{k+1} - d_k for k=0..17
        k = int(np.argmin(gaps))
        D = 0.5 * (d[k] + d[k + 1])
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        four = d[idx]
        ratio = np.maximum(four / D, D / four)
        raw = np.mean(ratio)
        scores[i] = 1.0 / raw
    return scores


def _vertex_score_clamped(polygon):
    raw = _vertex_score_raw(polygon)
    return np.minimum(1.0, raw)


def _metric(polygon):
    """The metric: minimum vertex score. Returns 0 if invalid."""
    n = polygon.shape[0]
    # validity: no repeated points, strictly convex (all turns same sign, nonzero), total turning +-2*pi
    # check repeated
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-9:
                return 0.0
    # compute turns
    edges = np.zeros((n, 2))
    for i in range(n):
        edges[i] = polygon[(i + 1) % n] - polygon[i]
    # cross products of consecutive edges
    turns = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        turns[i] = e1[0] * e2[1] - e1[1] * e2[0]
    # all turns must be nonzero and same sign
    if np.any(np.abs(turns) < 1e-12):
        return 0.0
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return 0.0
    # total turning: sum of signed angles
    angles = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-30)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        sin_a = turns[i] / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-30)
        angles[i] = np.arctan2(sin_a, cos_a)
    total = np.sum(angles)
    if abs(abs(total) - 2.0 * np.pi) > 1e-6:
        return 0.0
    # normalize
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists = np.linalg.norm(p, axis=1)
    mean_d = np.mean(dists)
    if mean_d <= 0.01:
        return 0.0
    p = p / mean_d
    # recompute metric on normalized
    vs = _vertex_score_clamped(p)
    return float(np.min(vs))


def _soft_min(scores, temperature=0.05):
    """Soft minimum via log-sum-exp: -logsumexp(-scores/temperature) * temperature."""
    s = scores / temperature
    m = np.min(s)
    return -temperature * (m + np.log(np.sum(np.exp(s - m))))


def _soft_metric(polygon, temperature=0.05):
    """Smooth surrogate for the metric (higher is better)."""
    n = polygon.shape[0]
    # basic validity check for smoothness (soft penalty for non-convex)
    edges = np.zeros((n, 2))
    for i in range(n):
        edges[i] = polygon[(i + 1) % n] - polygon[i]
    turns = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        turns[i] = e1[0] * e2[1] - e1[1] * e2[0]
    # soft penalty if any turn is near zero or mixed sign
    turn_penalty = 0.0
    min_turn = np.min(np.abs(turns))
    if min_turn < 1e-3:
        turn_penalty += (1e-3 - min_turn) ** 2
    # check sign consistency softly
    if np.any(turns > 0) and np.any(turns < 0):
        turn_penalty += 1.0
    # normalize
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists = np.linalg.norm(p, axis=1)
    mean_d = np.mean(dists)
    if mean_d <= 0.01:
        return -10.0
    p = p / mean_d
    raw_scores = _vertex_score_raw(p)
    # use soft min of the raw scores (before clamping) but clamp each to [0,1]
    clamped = np.minimum(1.0, raw_scores)
    sm = _soft_min(clamped, temperature)
    return sm - turn_penalty


def _fourier_to_polygon(params, n=20, base_radius=1.0):
    """Convert Fourier parameters to polygon vertices.
    params: array of [a1c, a1s, a2c, a2s, a3c, a3s, r1c, r1s, r2c, r2s]
    where angle deviation = sum_k (a_kc * cos(k*theta) + a_ks * sin(k*theta))
    and radius modulation = base_radius * (1 + sum_k (r_kc * cos(k*theta) + r_ks * sin(k*theta)))
    """
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta = base_angles
    a1c, a1s, a2c, a2s, a3c, a3s, r1c, r1s, r2c, r2s = params
    angle_dev = (a1c * np.cos(1 * theta) + a1s * np.sin(1 * theta)
                 + a2c * np.cos(2 * theta) + a2s * np.sin(2 * theta)
                 + a3c * np.cos(3 * theta) + a3s * np.sin(3 * theta))
    radius_mod = 1.0 + (r1c * np.cos(1 * theta) + r1s * np.sin(1 * theta)
                        + r2c * np.cos(2 * theta) + r2s * np.sin(2 * theta))
    angles = theta + angle_dev
    radii = base_radius * radius_mod
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _neg_soft_metric(params, temperature=0.05):
    polygon = _fourier_to_polygon(params)
    return -_soft_metric(polygon, temperature)


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0

    # Initial parameters: zero (regular polygon)
    x0 = np.zeros(10)

    # Try several initializations and temperatures
    best_params = x0.copy()
    best_score = _metric(_fourier_to_polygon(x0))

    temperatures = [0.1, 0.05, 0.02, 0.01]
    # Multiple random starts
    np.random.seed(42)
    for temp in temperatures:
        for trial in range(3):
            if time.time() > deadline - safety_margin:
                break
            if trial == 0:
                x0 = np.zeros(10)
            else:
                x0 = np.random.randn(10) * 0.05
            try:
                res = minimize(_neg_soft_metric, x0, args=(temp,), method='L-BFGS-B',
                               options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.success or res.fun < _neg_soft_metric(best_params, temp):
                    candidate = _fourier_to_polygon(res.x)
                    score = _metric(candidate)
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
            except Exception:
                pass
        if time.time() > deadline - safety_margin:
            break

    # Polish: refine with a tighter temperature and more iterations
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(_neg_soft_metric, best_params, args=(0.005,), method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-12})
            candidate = _fourier_to_polygon(res.x)
            score = _metric(candidate)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish: direct coordinate optimization from the best Fourier result
    if time.time() < deadline - safety_margin:
        polygon = _fourier_to_polygon(best_params)
        # Try a coordinate-level refinement with a soft metric
        def _neg_soft_coord(pflat, temperature=0.005):
            p = pflat.reshape(20, 2)
            return -_soft_metric(p, temperature)

        x0_coord = polygon.flatten()
        try:
            res = minimize(_neg_soft_coord, x0_coord, args=(0.005,), method='L-BFGS-B',
                           options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            candidate = res.x.reshape(20, 2)
            score = _metric(candidate)
            if score > best_score:
                best_score = score
                polygon = candidate
        except Exception:
            pass

    result = _fourier_to_polygon(best_params)
    # Ensure the result is valid; if not, fall back to regular polygon
    if _metric(result) <= 0.0:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        result = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return result


# EVOLVE-BLOCK-END
