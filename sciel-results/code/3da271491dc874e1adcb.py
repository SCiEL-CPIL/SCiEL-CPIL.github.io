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

def _angles_to_points(angles):
    n = len(angles)
    # Normalize angles to sum to 2*pi
    s = np.sum(angles)
    if s == 0:
        return np.zeros((n, 2))
    angles = angles / s * 2 * np.pi
    
    # Cumulative angles
    cum_angles = np.cumsum(angles)
    cum_angles = np.append([0], cum_angles)
    
    x = np.cos(cum_angles[:-1])
    y = np.sin(cum_angles[:-1])
    return np.stack([x, y], axis=1)

def _fourier_to_angles(params, n=20):
    """Convert Fourier mode coefficients to angle deviations.
    params: [a1, b1, a2, b2, a3, b3, a4, b4]
    Returns angles array of length n that sums to 2*pi.
    """
    base_angle = 2 * np.pi / n
    deviations = np.zeros(n)
    # Modes 1-4
    for k in range(1, 5):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        theta = 2 * np.pi * k * np.arange(n) / n
        deviations += a * np.cos(theta) + b * np.sin(theta)
    angles = base_angle + deviations
    # Ensure positive angles
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
    return _neg_score_from_angles(angles)

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: regular polygon
    angles0 = np.ones(n)
    best_points = _angles_to_points(angles0)
    best_score = _compute_score(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Fourier-based optimization with low-frequency modes
    # Use 8 parameters: 4 cosine + 4 sine coefficients for modes 1-4
    fourier_bounds = [(-0.5, 0.5)] * 8
    
    try:
        if time.time() < deadline - 180:
            result_fourier = differential_evolution(
                _neg_score_from_fourier, fourier_bounds, 
                seed=42, maxiter=200, tol=1e-8,
                polish=True, init='sobol',
                workers=1
            )
            if result_fourier.fun < -best_score:
                angles_opt = _fourier_to_angles(result_fourier.x, n)
                pts = _angles_to_points(angles_opt)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
    except Exception:
        pass
    
    # Phase 2: Full angle optimization from best Fourier result
    try:
        if time.time() < deadline - 120:
            # Get initial angles from best Fourier result if available
            if 'result_fourier' in locals() and result_fourier.fun < -best_score:
                init_angles = _fourier_to_angles(result_fourier.x, n)
            else:
                init_angles = angles0
            
            angle_bounds = [(0.05, 1.0)] * n  # Angles in units of 2*pi/n
            
            result_angles = differential_evolution(
                _neg_score_from_angles, angle_bounds, 
                seed=123, maxiter=100, tol=1e-7,
                polish=True, init='sobol',
                workers=1
            )
            if result_angles.fun < -best_score:
                angles_opt = result_angles.x
                pts = _angles_to_points(angles_opt)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
    except Exception:
        pass
    
    # Phase 3: Local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            # Use best angles from either phase
            if 'result_angles' in locals() and result_angles.fun < -best_score:
                init = result_angles.x
            elif 'result_fourier' in locals() and result_fourier.fun < -best_score:
                init = _fourier_to_angles(result_fourier.x, n)
            else:
                init = angles0
                
            result_local = minimize(_neg_score_from_angles, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            pts = _angles_to_points(result_local.x)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
            
    return best_points
