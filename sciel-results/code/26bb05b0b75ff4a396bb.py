"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices i, j, k
                # Using cross product formula: 0.5 * |cross(pj-pi, pk-pi)|
                cross = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                        (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _two_shell_points(params, n1, n2):
    """
    Generate points from two concentric regular polygons.
    params = [r1, r2, theta1, theta2]
    n1 points on circle of radius r1 starting at angle theta1
    n2 points on circle of radius r2 starting at angle theta2
    """
    r1, r2, theta1, theta2 = params
    angles1 = theta1 + 2 * np.pi * np.arange(n1) / n1
    angles2 = theta2 + 2 * np.pi * np.arange(n2) / n2
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    return np.vstack([pts1, pts2])


def _negative_score_two_shell(params, n1, n2):
    """Negative score for minimization."""
    points = _two_shell_points(params, n1, n2)
    return -_score(points)


def _optimize_partition(n1, n2, time_limit):
    """Optimize a two-shell configuration with n1 and n2 points."""
    start_time = time.time()
    
    # Multiple random starts
    best_params = None
    best_score = -np.inf
    
    n_starts = 30
    
    for s in range(n_starts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Random initial guess
        r1 = np.random.uniform(0.3, 1.0)
        r2 = np.random.uniform(0.3, 1.0)
        theta1 = np.random.uniform(0, 2 * np.pi)
        theta2 = np.random.uniform(0, 2 * np.pi)
        x0 = np.array([r1, r2, theta1, theta2])
        
        result = minimize(
            _negative_score_two_shell,
            x0,
            args=(n1, n2),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x
    
    if best_params is not None:
        best_points = _two_shell_points(best_params, n1, n2)
        return best_points, best_score
    
    return None, -np.inf


def construct():
    """
    Place 13 points to maximize the minimum triangle area / convex hull area.
    Uses a two-shell regular polygon structure with Nelder-Mead optimization.
    """
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Partitions of 13 into two parts
    partitions = [(1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]
    
    best_points = None
    best_score = -np.inf
    
    # Also try regular 13-gon as baseline
    angles = 2 * np.pi * np.arange(N) / N
    regular_points = np.column_stack((np.cos(angles), np.sin(angles)))
    regular_score = _score(regular_points)
    best_points = regular_points
    best_score = regular_score
    
    # Try each partition
    for n1, n2 in partitions:
        if time.time() - start_time > time_limit - 5:
            break
        
        points, score = _optimize_partition(n1, n2, time_limit - (time.time() - start_time))
        
        if points is not None and score > best_score:
            best_points = points
            best_score = score
    
    # Try some additional random perturbations of the best solution
    if time.time() - start_time < time_limit - 20 and best_points is not None:
        for _ in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
            pert_score = _score(perturbed)
            if pert_score > best_score:
                best_points = perturbed
                best_score = pert_score
    
    return best_points
