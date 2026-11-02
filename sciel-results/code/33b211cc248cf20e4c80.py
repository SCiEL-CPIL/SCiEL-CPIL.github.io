"""Heilbronn 13-point problem: symmetric hexagonal design + refinement."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 13

# EVOLVE-BLOCK-START

def _make_points(r, s, theta):
    """Build 13 points: 6 outer hexagon vertices + center + 6 inner hexagon vertices."""
    angles_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False) + theta
    outer = np.column_stack([r * np.cos(angles_outer), r * np.sin(angles_outer)])
    center = np.array([[0.0, 0.0]])
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + theta + np.pi / 6.0
    inner = np.column_stack([s * np.cos(angles_inner), s * np.sin(angles_inner)])
    return np.vstack([outer, center, inner])


def _min_triangle_area(pts):
    """Minimum area over all triples of points (using 2x area = |cross|)."""
    n = len(pts)
    min_2a = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pij = pts[j] - pi
            for k in range(j + 1, n):
                pik = pts[k] - pi
                cr = abs(pij[0] * pik[1] - pij[1] * pik[0])
                if cr < min_2a:
                    min_2a = cr
    return min_2a / 2.0


def _convex_hull_area(pts):
    """Area of the convex hull of pts (2D)."""
    # Use monotone chain
    pts_sorted = sorted(map(tuple, pts))
    if len(pts_sorted) <= 1:
        return 0.0
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _score_params(params):
    """Score = min_triangle_area / convex_hull_area. Return negative for minimization."""
    r, s, theta = params
    r = abs(r)
    s = abs(s)
    pts = _make_points(r, s, theta)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return -(min_area / hull_area)


def _score_points(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_area


def construct():
    deadline = time.time() + 115.0  # safety margin before 150s kill

    # Start from a good symmetric guess: regular hexagon with inner layer
    # The optimal for 13 points is known to be around the "hexagonal lattice" pattern.
    # Regular hexagon with 6 vertices + center + 6 mid-arc points is a classic candidate.
    # For a regular hexagon of circumradius R, area = (3*sqrt(3)/2)*R^2
    # Inner points at radius s with 60-degree offset.

    # Grid search over (r, s, theta) with a coarse grid first
    best_score = 0.0
    best_params = (1.0, 0.5, 0.0)

    # Coarse grid: r in [0.8, 1.2], s in [0.1, 0.9], theta in [0, pi/3]
    r_vals = np.linspace(0.7, 1.3, 13)
    s_vals = np.linspace(0.1, 0.95, 17)
    theta_vals = np.linspace(0.0, np.pi / 3.0, 7)

    for r in r_vals:
        for s in s_vals:
            for th in theta_vals:
                params = (r, s, th)
                neg_score = _score_params(params)
                score = -neg_score
                if score > best_score:
                    best_score = score
                    best_params = params

    if time.time() > deadline - 30:
        pts = _make_points(*best_params)
        return pts

    # Refine with local optimization (Nelder-Mead) from the best grid point
    try:
        result = minimize(
            _score_params,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x
    except Exception:
        pass

    if time.time() > deadline - 15:
        pts = _make_points(*best_params)
        return pts

    # Try a few additional local refinements from nearby points
    for perturb in [(0.02, 0.02, 0.05), (-0.02, 0.02, 0.0), (0.0, -0.03, 0.1), (0.03, -0.01, -0.05)]:
        if time.time() > deadline - 10:
            break
        start = np.array(best_params) + np.array(perturb)
        try:
            result = minimize(
                _score_params,
                start,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_params = result.x
        except Exception:
            pass

    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _score_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_params = result.x
        except Exception:
            pass

    pts = _make_points(*best_params)
    return pts

# EVOLVE-BLOCK-END
