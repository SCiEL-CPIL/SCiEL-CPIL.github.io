"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        if len(points) < 3:
            return 0.0
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _two_shell_params_to_points(params):
    """
    Convert 2D parameter vector [r_ratio, rotation] to 13 points.
    Shell 1: 2 points on unit circle (outer, convex hull)
    Shell 2: 11 points on circle of radius r_ratio, rotated by 'rotation'
    """
    r = params[0]
    rot = params[1]
    
    # Outer shell: 2 points
    outer_angles = np.array([0, np.pi])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    # Inner shell: 11 points
    inner_angles = np.linspace(0, 2 * np.pi, 11, endpoint=False) + rot
    inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])


def _two_shell_objective(params):
    """Objective: minimize negative score for 2-shell configuration."""
    r = params[0]
    if r < 0.01 or r > 0.99:
        return 1e10
    rot = params[1]
    
    points = _two_shell_params_to_points(params)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(points)
    return -min_area / hull_area


def _three_shell_params_to_points(params):
    """
    Convert 3D parameter vector [r1, r2, rotation] to 13 points.
    Shell 1: 3 points on unit circle (outer)
    Shell 2: 5 points on circle of radius r1
    Shell 3: 5 points on circle of radius r2, rotated
    """
    r1 = params[0]
    r2 = params[1]
    rot = params[2]
    
    # Outer shell: 3 points
    outer_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    # Middle shell: 5 points
    mid_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rot * 0.5
    mid_pts = r1 * np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])
    
    # Inner shell: 5 points
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rot
    inner_pts = r2 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, mid_pts, inner_pts])


def _three_shell_objective(params):
    """Objective: minimize negative score for 3-shell configuration."""
    r1, r2, rot = params
    if r1 < 0.01 or r1 > 0.99 or r2 < 0.01 or r2 > 0.99:
        return 1e10
    
    points = _three_shell_params_to_points(params)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(points)
    return -min_area / hull_area


def _four_shell_params_to_points(params):
    """
    Convert 4D parameter vector [r1, r2, r3, rotation] to 13 points.
    Shell 1: 4 points on unit circle (outer)
    Shell 2: 3 points on circle of radius r1
    Shell 3: 3 points on circle of radius r2
    Shell 4: 3 points on circle of radius r3
    """
    r1, r2, r3, rot = params
    
    # Outer shell: 4 points
    outer_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    # Shell 2: 3 points
    s2_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False) + rot * 0.25
    s2_pts = r1 * np.column_stack([np.cos(s2_angles), np.sin(s2_angles)])
    
    # Shell 3: 3 points
    s3_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False) + rot * 0.5
    s3_pts = r2 * np.column_stack([np.cos(s3_angles), np.sin(s3_angles)])
    
    # Shell 4: 3 points
    s4_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False) + rot * 0.75
    s4_pts = r3 * np.column_stack([np.cos(s4_angles), np.sin(s4_angles)])
    
    return np.vstack([outer_pts, s2_pts, s3_pts, s4_pts])


def _four_shell_objective(params):
    """Objective: minimize negative score for 4-shell configuration."""
    r1, r2, r3, rot = params
    if min(r1, r2, r3) < 0.01 or max(r1, r2, r3) > 0.99:
        return 1e10
    
    points = _four_shell_params_to_points(params)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(points)
    return -min_area / hull_area


def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_points = _regular_polygon(N)
    best_score = _score(best_points)

    rng = np.random.default_rng(42)

    # Phase 1: Optimize 2-shell structure (2 outer + 11 inner)
    # Parameters: [radius_ratio, rotation]
    bounds_2shell = [(0.1, 0.95), (-np.pi, np.pi)]
    
    for restart in range(20):
        if time.time() > deadline:
            break

        if restart == 0:
            x0 = np.array([0.5, 0.0])
        else:
            x0 = np.array([rng.uniform(0.2, 0.9), rng.uniform(-np.pi, np.pi)])

        res = minimize(_two_shell_objective, x0, method='SLSQP',
                       bounds=bounds_2shell,
                       options={'maxiter': 200, 'ftol': 1e-12})

        if time.time() > deadline:
            break

        candidate = _two_shell_params_to_points(res.x)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    # Phase 2: Optimize 3-shell structure (3 outer + 5 middle + 5 inner)
    bounds_3shell = [(0.1, 0.95), (0.1, 0.95), (-np.pi, np.pi)]
    
    for restart in range(15):
        if time.time() > deadline:
            break

        if restart == 0:
            x0 = np.array([0.6, 0.3, 0.0])
        else:
            x0 = np.array([rng.uniform(0.2, 0.9), rng.uniform(0.1, 0.8), rng.uniform(-np.pi, np.pi)])

        res = minimize(_three_shell_objective, x0, method='SLSQP',
                       bounds=bounds_3shell,
                       options={'maxiter': 200, 'ftol': 1e-12})

        if time.time() > deadline:
            break

        candidate = _three_shell_params_to_points(res.x)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    # Phase 3: Optimize 4-shell structure (4 outer + 3+3+3 inner)
    bounds_4shell = [(0.1, 0.95), (0.1, 0.95), (0.1, 0.95), (-np.pi, np.pi)]
    
    for restart in range(10):
        if time.time() > deadline:
            break

        if restart == 0:
            x0 = np.array([0.7, 0.5, 0.3, 0.0])
        else:
            x0 = np.array([rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.8), 
                          rng.uniform(0.1, 0.7), rng.uniform(-np.pi, np.pi)])

        res = minimize(_four_shell_objective, x0, method='SLSQP',
                       bounds=bounds_4shell,
                       options={'maxiter': 200, 'ftol': 1e-12})

        if time.time() > deadline:
            break

        candidate = _four_shell_params_to_points(res.x)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    # Phase 4: Full 26D polish from best symmetric configuration
    def objective_full(x):
        pts = x.reshape(-1, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area

    for restart in range(5):
        if time.time() > deadline:
            break

        if restart == 0:
            points0 = best_points.copy()
        else:
            points0 = best_points + rng.normal(0, 0.01, size=best_points.shape)

        res = minimize(objective_full, points0.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})

        if time.time() > deadline:
            break

        candidate = res.x.reshape(-1, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    return best_points
# EVOLVE-BLOCK-END
