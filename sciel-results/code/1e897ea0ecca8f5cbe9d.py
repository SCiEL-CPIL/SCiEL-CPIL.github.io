# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_TIME_LIMIT = 600.0  # seconds; leave safety margin
_N = 20
_NUM_PARAMS = 10  # 5 cosine modes x 2 (amplitude + phase)


def _vertex_score(d):
    """Compute the per-vertex score given sorted distances d (length 19)."""
    if d[0] < 0.05:
        return 0.0
    diffs = d[1:] - d[:-1]
    k = int(np.argmin(diffs))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = np.array([0, 1, 2, 3])
    elif k == 17:
        idx = np.array([15, 16, 17, 18])
    else:
        idx = np.array([k - 1, k, k + 1, k + 2])
    four = d[idx]
    raw = np.mean(np.maximum(four / D, D / four))
    return min(1.0, 1.0 / raw)


def _score_polygon(pts):
    """
    Score a (20,2) array of vertices. Returns (score, is_valid).
    Normalizes: subtract centroid, divide by mean distance to centroid.
    Checks convexity. Computes min vertex score.
    """
    if pts.shape != (20, 2):
        return 0.0, False
    if not np.all(np.isfinite(pts)):
        return 0.0, False

    centroid = pts.mean(axis=0)
    p = pts - centroid
    dists_to_centroid = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0, False
    p = p / mean_dist

    # Check no repeated points
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return 0.0, False

    # Check strictly convex: all cross products same sign, nonzero
    cross_signs = []
    for i in range(20):
        a = p[i]
        b = p[(i + 1) % 20]
        c = p[(i + 2) % 20]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-12:
            return 0.0, False
        cross_signs.append(np.sign(cross))
    if len(set(cross_signs)) != 1:
        return 0.0, False

    # Compute per-vertex scores
    scores = []
    for i in range(20):
        d = np.sort(np.linalg.norm(p - p[i], axis=1))
        d = d[1:]  # exclude self
        scores.append(_vertex_score(d))

    min_score = min(scores)
    return min_score, True


def _build_polygon_radial(params):
    """
    Build polygon from radial cosine series parameters.
    params: array of 10 values: [a1, p1, a2, p2, a3, p3, a4, p4, a5, p5]
    r(theta) = 1 + sum_k a_k * cos(k*theta + p_k)
    theta_i = 2*pi*i/20
    """
    i = np.arange(20)
    theta = 2.0 * np.pi * i / 20.0
    r = np.ones(20)
    for k in range(5):
        a_k = params[2 * k]
        p_k = params[2 * k + 1]
        r += a_k * np.cos(k * theta + p_k)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def _soft_min(scores, temperature=0.01):
    """Smooth approximation of min using log-sum-exp."""
    scores = np.asarray(scores, dtype=np.float64)
    smin = scores.min()
    # Soft min: -1/T * log(sum(exp(-T*s)))  approximates min as T->0
    shifted = -scores / temperature
    max_shifted = shifted.max()
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -temperature * log_sum


def _objective_smooth(params, temperature=0.01):
    """Minimize negative soft-min score (smoother landscape)."""
    pts = _build_polygon_radial(params)
    score, valid = _score_polygon(pts)
    if not valid:
        return 1.0
    # Recompute individual scores for soft-min
    centroid = pts.mean(axis=0)
    p = pts - centroid
    dists_to_centroid = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 1.0
    p = p / mean_dist

    scores = []
    for i in range(20):
        d = np.sort(np.linalg.norm(p - p[i], axis=1))
        d = d[1:]
        scores.append(_vertex_score(d))
    soft = _soft_min(scores, temperature)
    return -soft


def _objective(params):
    """Minimize negative exact score (with validity penalty)."""
    pts = _build_polygon_radial(params)
    score, valid = _score_polygon(pts)
    if not valid:
        return 1.0
    return -score


def _regular_polygon():
    """Fallback: regular 20-gon."""
    i = np.arange(20)
    angles = 2.0 * np.pi * i / 20.0
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum vertex equidistance score.
    Uses radial cosine series parameterization with differential evolution.
    """
    # Bounds: amplitudes small to ensure convexity, phases free
    bounds = []
    for k in range(5):
        bounds.append((-0.08, 0.08))  # amplitude
        bounds.append((-np.pi, np.pi))  # phase

    # Initial best: regular polygon (all amplitudes = 0)
    best_pts = _regular_polygon()
    best_score, _ = _score_polygon(best_pts)
    best_params = np.zeros(_NUM_PARAMS)

    elapsed = time.time() - _START_TIME

    # Phase 1: DE with soft-min objective (smoother landscape)
    if elapsed < _TIME_LIMIT - 120:
        try:
            result = differential_evolution(
                lambda p: _objective_smooth(p, temperature=0.02),
                bounds,
                maxiter=150,
                popsize=12,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol',
                updating='immediate',
                maxfev=15000,
            )
            if result.fun < 0.5:
                candidate_pts = _build_polygon_radial(result.x)
                cand_score, cand_valid = _score_polygon(candidate_pts)
                if cand_valid and cand_score > best_score:
                    best_pts = candidate_pts
                    best_score = cand_score
                    best_params = result.x.copy()
        except Exception:
            pass

    elapsed = time.time() - _START_TIME

    # Phase 2: DE with exact objective (sharper)
    if elapsed < _TIME_LIMIT - 80:
        try:
            result2 = differential_evolution(
                _objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                seed=123,
                polish=False,
                init='sobol',
                updating='immediate',
                maxfev=20000,
            )
            if result2.fun < 0.5:
                candidate_pts = _build_polygon_radial(result2.x)
                cand_score, cand_valid = _score_polygon(candidate_pts)
                if cand_valid and cand_score > best_score:
                    best_pts = candidate_pts
                    best_score = cand_score
                    best_params = result2.x.copy()
        except Exception:
            pass

    elapsed = time.time() - _START_TIME

    # Phase 3: Multi-start Nelder-Mead from best params
    if elapsed < _TIME_LIMIT - 40:
        try:
            seeds = [best_params]
            # Add a few perturbations
            for _ in range(3):
                pert = best_params + np.random.normal(0, 0.02, _NUM_PARAMS)
                # Clip amplitudes
                for k in range(5):
                    pert[2*k] = np.clip(pert[2*k], -0.08, 0.08)
                    pert[2*k+1] = np.arctan2(np.sin(pert[2*k+1]), np.cos(pert[2*k+1]))
                seeds.append(pert)

            for x0 in seeds:
                if time.time() - _START_TIME > _TIME_LIMIT - 20:
                    break
                res = minimize(
                    _objective, x0, method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-9,
                             'adaptive': True}
                )
                if res.fun < 0.5:
                    candidate_pts = _build_polygon_radial(res.x)
                    cand_score, cand_valid = _score_polygon(candidate_pts)
                    if cand_valid and cand_score > best_score:
                        best_pts = candidate_pts
                        best_score = cand_score
                        best_params = res.x.copy()
        except Exception:
            pass

    # Final time check
    if time.time() - _START_TIME > _TIME_LIMIT - 10:
        return best_pts

    return best_pts


# EVOLVE-BLOCK-END
