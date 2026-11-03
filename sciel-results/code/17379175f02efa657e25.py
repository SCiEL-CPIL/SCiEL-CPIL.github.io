"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START

def _triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

def _all_triangle_areas(points):
    """Compute all triangle areas for n points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e6
    except Exception:
        return -1e6
    
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    if min_area < 1e-9:
        return -1e6
    
    return min_area / hull_area

def _soft_min(areas, beta=500.0):
    """Soft minimum using log-sum-exp trick."""
    # For minimization: -logsumexp(-x/beta) * beta
    # Use stable version
    min_val = np.min(areas)
    shifted = -areas / beta + min_val / beta
    logsumexp = np.log(np.sum(np.exp(shifted)))
    return -(beta * logsumexp) - min_val

def _negative_score_soft(points, beta=500.0):
    """Negative soft-min score for minimization."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e6
    except Exception:
        return 1e6
    
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    areas = np.array(areas)
    
    if np.min(areas) < 1e-9:
        return 1e6
    
    soft_min_val = _soft_min(areas, beta)
    return -soft_min_val / hull_area

def _regular_n_gon(n, radius=1.0, angle_offset=0.0):
    """Generate vertices of a regular n-gon centered at origin."""
    angles = angle_offset + 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _optimize_from_start(points0, max_time=100.0):
    """Optimize from a starting configuration using Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # Flatten points
    x0 = points0.flatten()
    
    best_x = x0.copy()
    best_score = _score(points0)
    
    # Use Nelder-Mead with the negative soft-min score
    result = minimize(
        _negative_score_soft,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    # Check time
    if time.time() > deadline:
        return best_x.reshape(-1, 2), best_score
    
    x_opt = result.x.reshape(-1, 2)
    score_opt = _score(x_opt)
    
    if score_opt > best_score:
        best_x = x_opt
        best_score = score_opt
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -1e6
    
    # Strategy: Start from regular 14-gon and optimize
    # The regular polygon is a good symmetric starting point
    
    # Try multiple starting configurations
    starting_configs = []
    
    # Regular 14-gon
    starting_configs.append(_regular_n_gon(N, radius=1.0, angle_offset=0.0))
    starting_configs.append(_regular_n_gon(N, radius=1.0, angle_offset=np.pi / N))
    
    # Slightly perturbed regular 14-gons
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _regular_n_gon(N, radius=1.0, angle_offset=0.0)
        perturbed = base + 0.01 * rng.randn(N, 2)
        starting_configs.append(perturbed)
    
    # Two concentric 7-gons (inner and outer)
    outer_7 = _regular_n_gon(7, radius=1.2, angle_offset=0.0)
    inner_7 = _regular_n_gon(7, radius=0.5, angle_offset=np.pi / 7)
    two_ring = np.vstack([outer_7, inner_7])
    starting_configs.append(two_ring)
    
    # Regular 14-gon with alternating radii
    angles = 2 * np.pi * np.arange(N) / N
    radii = np.ones(N)
    radii[::2] = 1.1
    radii[1::2] = 0.9
    alt_radii = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    starting_configs.append(alt_radii)
    
    # Optimize from each starting configuration
    for i, start in enumerate(starting_configs):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time based on remaining budget
        time_per_opt = min(remaining / max(1, len(starting_configs) - i), 40.0)
        
        points_opt, score_opt = _optimize_from_start(start, max_time=time_per_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt.copy()
    
    # If we still have time, do additional refinement on the best solution
    if best_points is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 10.0
        points_refined, score_refined = _optimize_from_start(best_points, max_time=remaining)
        if score_refined > best_score:
            best_points = points_refined
            best_score = score_refined
    
    if best_points is None:
        # Fallback: regular 14-gon
        best_points = _regular_n_gon(N, radius=1.0, angle_offset=0.0)
    
    # Final check
    final_score = _score(best_points)
    if final_score < 0:
        best_points = _regular_n_gon(N, radius=1.0, angle_offset=0.0)
    
    return best_points

# EVOLVE-BLOCK-END
