import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

def _compute_score(points):
    """Compute the minimum vertex score for a given polygon."""
    n = len(points)
    # Normalize: subtract centroid, divide by mean distance to centroid
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

def _fourier_to_points(params, n=20):
    """Convert Fourier mode coefficients to polygon vertices.
    params: [a1, b1, a2, b2, a3, b3, a4, b4]
    Returns (n, 2) array of points.
    """
    base_angle = 2 * np.pi / n
    angles = np.full(n, base_angle)
    # Modes 1-4
    for k in range(1, 5):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        theta = 2 * np.pi * k * np.arange(n) / n
        angles += a * np.cos(theta) + b * np.sin(theta)
    # Ensure positive angles and normalize
    angles = np.maximum(angles, 0.1)
    angles = angles / np.sum(angles) * 2 * np.pi
    
    # Build polygon from cumulative angles
    cum_angles = np.cumsum(angles)
    cum_angles = np.append([0], cum_angles)
    
    x = np.cos(cum_angles[:-1])
    y = np.sin(cum_angles[:-1])
    return np.stack([x, y], axis=1)

def _soft_min_vertex_score(points, beta=50.0):
    """Compute a smooth approximation of the minimum vertex score using log-sum-exp.
    Higher beta = closer to hard min.
    """
    n = len(points)
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Check convexity
    edges = np.roll(pts, -1, axis=0) - pts
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    
    # Compute per-vertex scores
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
    
    # Soft min: -log(sum(exp(-beta * scores))) / beta
    # This is smooth and approximates min(scores) as beta -> inf
    if np.any(scores <= 0):
        return 0.0
    neg_scores = -beta * scores
    # Use log-sum-exp trick for numerical stability
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min = -log_sum / beta
    return soft_min

def _neg_soft_score_from_fourier(params, beta=50.0):
    """Negative soft-min score for optimization (minimize negative = maximize score)."""
    pts = _fourier_to_points(params)
    return -_soft_min_vertex_score(pts, beta)

def _neg_score_from_fourier(params):
    """Negative hard score for final evaluation."""
    pts = _fourier_to_points(params)
    return -_compute_score(pts)

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: regular polygon (all Fourier coefficients zero)
    params0 = np.zeros(8)
    best_points = _fourier_to_points(params0)
    best_score = _compute_score(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Differential evolution on 8 Fourier parameters
    bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.15, 0.15), (-0.15, 0.15),
              (-0.08, 0.08), (-0.08, 0.08), (-0.04, 0.04), (-0.04, 0.04)]
    
    try:
        if time.time() < deadline - 120:
            result_de = differential_evolution(
                _neg_soft_score_from_fourier, bounds,
                args=(50.0,),
                seed=42, maxiter=200, tol=1e-8,
                polish=True, init='sobol',
                workers=1
            )
            if result_de.fun < -best_score:
                params_opt = result_de.x
                pts = _fourier_to_points(params_opt)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
    except Exception:
        pass
    
    # Phase 2: Multiple local refinements from different starting points
    if time.time() < deadline - 60:
        # Try several different initializations
        init_candidates = [params0]
        if 'result_de' in locals() and result_de.fun < -best_score:
            init_candidates.append(result_de.x)
        
        # Add some random perturbations
        for trial in range(5):
            if time.time() > deadline - 40:
                break
            random_params = np.random.uniform(-0.1, 0.1, size=8)
            init_candidates.append(random_params)
        
        for init in init_candidates:
            if time.time() > deadline - 30:
                break
            try:
                # First pass with moderate beta
                result_local = minimize(
                    _neg_soft_score_from_fourier, init,
                    args=(30.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                params_local = result_local.x
                pts = _fourier_to_points(params_local)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            # Use the best params found so far
            if 'result_de' in locals() and result_de.fun < -best_score:
                init_final = result_de.x
            else:
                init_final = params0
            
            # Multiple polish passes with increasing beta
            for beta in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 5:
                    break
                result_polish = minimize(
                    _neg_soft_score_from_fourier, init_final,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                init_final = result_polish.x
                pts = _fourier_to_points(init_final)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish on the hard objective
    if time.time() < deadline - 5:
        try:
            if 'init_final' in locals():
                init_nm = init_final
            else:
                init_nm = params0
            
            result_nm = minimize(
                _neg_score_from_fourier, init_nm,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            pts = _fourier_to_points(result_nm.x)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
    
    return best_points
