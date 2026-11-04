"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286
TRI_ARR = np.array(TRI_INDICES, dtype=np.int32)  # (286, 3)

def _all_tri_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    i, j, k = TRI_ARR[:,0], TRI_ARR[:,1], TRI_ARR[:,2]
    a = pts[i]
    b = pts[j]
    c = pts[k]
    # 0.5 * |cross(b-a, c-a)|
    v1 = b - a
    v2 = c - a
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _fourier_to_points(fourier_params, n_points=N):
    """
    Convert Fourier series parameters to points on a smooth convex curve.
    
    Parameters:
    - fourier_params: array of shape (1 + 2*n_modes,)
      [r0, a1, b1, a2, b2, ..., an, bn]
      where r0 is the mean radius, and (an, bn) are Fourier coefficients
      for mode n: r(θ) = r0 + Σ (an*cos(n*θ) + bn*sin(n*θ))
    
    Returns:
    - points: array of shape (n_points, 2)
    """
    n_modes = (len(fourier_params) - 1) // 2
    r0 = fourier_params[0]
    
    # Angles for n_points equally spaced
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Compute radius at each angle
    radii = np.full(n_points, r0)
    for n in range(1, n_modes + 1):
        a_n = fourier_params[1 + 2*(n-1)]
        b_n = fourier_params[2 + 2*(n-1)]
        radii += a_n * np.cos(n * angles) + b_n * np.sin(n * angles)
    
    # Ensure positive radii (clip to avoid negative values)
    radii = np.maximum(radii, 0.1)
    
    # Convert to Cartesian
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return points

def _points_to_fourier(points, n_modes=4):
    """
    Fit Fourier series to given points to get initial parameters.
    Uses least squares to fit r(θ) = r0 + Σ (an*cos(n*θ) + bn*sin(n*θ))
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Build design matrix
    # Columns: [1, cos(θ), sin(θ), cos(2θ), sin(2θ), ...]
    A = np.ones((N, 1 + 2*n_modes))
    for n in range(1, n_modes + 1):
        A[:, 1 + 2*(n-1)] = np.cos(n * angles)
        A[:, 2 + 2*(n-1)] = np.sin(n * angles)
    
    # Least squares fit
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, radii, rcond=None)
    except:
        coeffs = np.zeros(1 + 2*n_modes)
        coeffs[0] = np.mean(radii)
    
    return coeffs

def _negative_score_from_fourier(fourier_params):
    """Objective for differential evolution: negative score from Fourier params."""
    try:
        points = _fourier_to_points(fourier_params)
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 1e6
        score = _score(points)
        return -score
    except:
        return 1e6

def _refine_cartesian(points, time_budget, rng):
    """
    Refine points using local optimization (Nelder-Mead + perturbation).
    """
    start_time = time.time()
    best_x = points.flatten().copy()
    best_score = _score(points)
    
    # Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_x, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                         'initial_simplex': 1e-3}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _score(cand.reshape(N, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
        except Exception:
            pass
    
    # Local perturbation search
    if time.time() - start_time < time_budget - 1:
        best_pts = best_x.reshape(N, 2)
        for _ in range(30):
            if time.time() - start_time > time_budget - 0.5:
                break
            areas = _all_tri_areas(best_pts)
            min_idx = np.argmin(areas)
            i, j, k = TRI_INDICES[min_idx]
            
            for pidx in [i, j, k]:
                for scale in [0.01, 0.005, 0.002]:
                    if time.time() - start_time > time_budget - 0.3:
                        break
                    trial = best_pts.copy()
                    trial[pidx] += rng.normal(0, scale, size=2)
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_x = trial.flatten().copy()
                        best_pts = trial
                        break
    
    return best_x.reshape(N, 2), best_score

def construct():
    """
    Main entry point: find optimal 13-point configuration.
    
    Strategy:
    1. Use differential evolution on low-dimensional Fourier parameters
       to find a good smooth convex shape
    2. Refine with local Cartesian optimization
    """
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    # Number of Fourier modes to optimize
    n_modes = 4
    n_params = 1 + 2 * n_modes  # r0 + (a1,b1) + (a2,b2) + (a3,b3) + (a4,b4)
    
    # Bounds for Fourier parameters
    # r0: mean radius, should be positive and reasonable
    # Higher modes: smaller amplitudes to maintain smoothness
    bounds = [(0.3, 2.0)]  # r0
    for n in range(1, n_modes + 1):
        # Amplitude decreases with mode number
        amp_bound = 0.3 / n
        bounds.extend([(-amp_bound, amp_bound), (-amp_bound, amp_bound)])
    
    # Phase 1: Differential evolution on Fourier parameters
    de_time_budget = 100.0
    de_start = time.time()
    
    try:
        de_result = differential_evolution(
            _negative_score_from_fourier,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=False,  # We'll do our own polishing
            updating='deferred',
            workers=1
        )
        
        if de_result.fun < 1e5:
            best_fourier = de_result.x
            best_points = _fourier_to_points(best_fourier)
            best_score = _score(best_points)
        else:
            # Fallback: regular polygon
            best_points = _fourier_to_points(np.array([1.0] + [0.0]*(2*n_modes)))
            best_score = _score(best_points)
    except Exception:
        # Fallback: regular polygon
        best_points = _fourier_to_points(np.array([1.0] + [0.0]*(2*n_modes)))
        best_score = _score(best_points)
    
    # Phase 2: Refine with local Cartesian optimization
    remaining_time = total_budget - (time.time() - start_time) - 3.0
    if remaining_time > 5.0:
        try:
            refined_points, refined_score = _refine_cartesian(
                best_points, remaining_time, rng
            )
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
        except Exception:
            pass
    
    # Final check: ensure valid output
    if best_points is None or np.any(np.isnan(best_points)) or np.any(np.isinf(best_points)):
        # Ultimate fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
