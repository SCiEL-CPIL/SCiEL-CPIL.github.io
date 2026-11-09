import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """Compute the minimum vertex score for a given polygon."""
    n = len(points)
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Check strict convexity
    edges = np.roll(pts, -1, axis=0) - pts
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d_sorted = np.sort(np.delete(d, i))
        if np.min(d_sorted) < 0.05:
            min_score = 0.0
            break
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        if k == 0:
            four = d_sorted[0:4]
        elif k == 17:
            four = d_sorted[15:19]
        else:
            four = d_sorted[k-1:k+3]
        ratios = np.maximum(four / D, D / four)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
    return min_score

def _angles_to_points(angles):
    n = len(angles)
    s = np.sum(angles)
    if s == 0:
        return np.zeros((n, 2))
    angles = angles / s * 2 * np.pi
    cum_angles = np.cumsum(angles)
    cum_angles = np.append([0], cum_angles)
    x = np.cos(cum_angles[:-1])
    y = np.sin(cum_angles[:-1])
    return np.stack([x, y], axis=1)

def _fourier_to_angles(params, n=20):
    """Convert Fourier mode coefficients to angle deviations."""
    base_angle = 2 * np.pi / n
    deviations = np.zeros(n)
    for k in range(1, 5):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        theta = 2 * np.pi * k * np.arange(n) / n
        deviations += a * np.cos(theta) + b * np.sin(theta)
    angles = base_angle + deviations
    angles = np.maximum(angles, 0.05)
    return angles

def _score_from_angles(angles):
    pts = _angles_to_points(angles)
    return _compute_score(pts)

def _neg_score_from_angles(angles):
    return -_score_from_angles(angles)

def _neg_score_from_fourier(params):
    n = 20
    angles = _fourier_to_angles(params, n)
    return -_score_from_angles(angles)

def _soft_neg_score_from_angles(angles, beta=30.0):
    """Smooth approximation of negative min vertex score for better optimization landscape."""
    n = len(angles)
    pts = _angles_to_points(angles)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 10.0
    pts = pts / mean_dist
    
    edges = np.roll(pts, -1, axis=0) - pts
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if np.any(np.abs(cross) < 1e-12):
        return 10.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 10.0
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d_sorted = np.sort(np.delete(d, i))
        if np.min(d_sorted) < 0.05:
            scores[i] = 0.0
            continue
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            four = d_sorted[0:4]
        elif k == 17:
            four = d_sorted[15:19]
        else:
            four = d_sorted[k-1:k+3]
        ratios = np.maximum(four / D, D / four)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    
    # Soft min: -log(sum(exp(-beta*s))) / beta
    # This is smooth and differentiable
    if np.any(scores == 0):
        return 10.0
    soft_min = -np.log(np.sum(np.exp(-beta * scores))) / beta
    return -soft_min

def _coordinate_descent_refine(angles, max_iter=200, tol=1e-10):
    """Refine angles via coordinate descent on individual angles."""
    n = len(angles)
    best = angles.copy()
    best_score = _score_from_angles(best)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            # Try perturbing angle i
            for delta in [0.001, -0.001, 0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                trial = best.copy()
                trial[i] = max(0.05, best[i] + delta)
                s = _score_from_angles(trial)
                if s > best_score + 1e-12:
                    best = trial
                    best_score = s
                    improved = True
                    break
        if not improved:
            break
    return best, best_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial guess: regular polygon
    angles0 = np.ones(n) * (2 * np.pi / n)
    best_points = _angles_to_points(angles0)
    best_score = _compute_score(best_points)
    
    np.random.seed(42)
    best_angles = angles0.copy()
    
    # Phase 1: Multi-start Fourier-based optimization
    # Try several different DE seeds on the 8-parameter Fourier space
    fourier_bounds = [(-0.4, 0.4)] * 8
    fourier_seeds = [42, 123, 456, 789]
    
    for seed_idx, seed in enumerate(fourier_seeds):
        if time.time() > deadline - 200:
            break
        try:
            result_fourier = differential_evolution(
                _neg_score_from_fourier, fourier_bounds,
                seed=seed, maxiter=150, tol=1e-8,
                polish=True, init='sobol',
                workers=1
            )
            angles_opt = _fourier_to_angles(result_fourier.x, n)
            pts = _angles_to_points(angles_opt)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                best_angles = angles_opt
        except Exception:
            pass
    
    # Phase 2: Soft-score optimization on full angles from best Fourier result
    if time.time() < deadline - 100:
        try:
            # Use soft score for smoother landscape
            result_soft = differential_evolution(
                lambda a: _soft_neg_score_from_angles(a, beta=20.0),
                [(0.1, 1.0)] * n,
                seed=42, maxiter=80, tol=1e-7,
                polish=True, init='sobol',
                workers=1
            )
            pts = _angles_to_points(result_soft.x)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                best_angles = result_soft.x
        except Exception:
            pass
    
    # Phase 3: Full angle DE from best angles
    if time.time() < deadline - 60:
        try:
            result_angles = differential_evolution(
                _neg_score_from_angles,
                [(0.05, 1.0)] * n,
                seed=123, maxiter=60, tol=1e-7,
                polish=True, init='sobol',
                workers=1
            )
            pts = _angles_to_points(result_angles.x)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                best_angles = result_angles.x
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead refinement
    if time.time() < deadline - 30:
        try:
            result_local = minimize(
                _neg_score_from_angles, best_angles,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            pts = _angles_to_points(result_local.x)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                best_angles = result_local.x
        except Exception:
            pass
    
    # Phase 5: Coordinate descent final refinement
    if time.time() < deadline - 10:
        try:
            refined_angles, refined_score = _coordinate_descent_refine(best_angles, max_iter=100)
            if refined_score > best_score:
                best_score = refined_score
                best_points = _angles_to_points(refined_angles)
        except Exception:
            pass
    
    return best_points
