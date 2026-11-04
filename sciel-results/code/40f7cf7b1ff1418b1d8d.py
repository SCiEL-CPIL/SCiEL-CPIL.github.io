"""Rewritten program for Heilbronn convex regions n=14."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 < n:
                v_ji = pts[j] - pts[i]
                v_ki = pts[j+1:] - pts[i]
                # 2D cross product (determinant)
                dets = v_ji[0] * v_ki[:, 1] - v_ji[1] * v_ki[:, 0]
                areas = 0.5 * np.abs(dets)
                min_in_loop = np.min(areas)
                if min_in_loop < min_a:
                    min_a = min_in_loop
                    if min_a < 1e-12:
                        return 0.0
    return min_a

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    cha = _convex_hull_area(pts)
    if cha < 1e-12:
        return 0.0
    mta = _min_triangle_area(pts)
    return mta / cha

def _fourier_points(coeffs):
    """
    Generate 14 points using Fourier series for radius.
    coeffs: [c1, c2, c3]
    r(theta) = 1 + c1*cos(theta) + c2*cos(2*theta) + c3*cos(3*theta)
    """
    thetas = 2 * np.pi * np.arange(N) / N
    c1, c2, c3 = coeffs
    r = 1.0 + c1 * np.cos(thetas) + c2 * np.cos(2 * thetas) + c3 * np.cos(3 * thetas)
    # Ensure positive radius
    r = np.maximum(r, 0.1)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack((x, y))

def _objective(coeffs):
    """Minimize negative score."""
    pts = _fourier_points(coeffs)
    s = _score(pts)
    return -s

def construct():
    t_start = time.time()
    deadline = t_start + 110.0

    # Initial guess: circle
    best_coeffs = np.zeros(3)
    best_score = _score(_fourier_points(best_coeffs))
    
    # Optimization
    bounds = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]
    
    try:
        if time.time() < deadline - 30.0:
            res = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                polish=True,
                workers=1
            )
            if res.fun < -best_score:
                best_coeffs = res.x
                best_score = -res.fun
    except Exception:
        pass

    # Local polish with L-BFGS-B
    try:
        if time.time() < deadline - 10.0:
            res = minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < -best_score:
                best_coeffs = res.x
                best_score = -res.fun
    except Exception:
        pass

    return _fourier_points(best_coeffs)

# EVOLVE-BLOCK-END
