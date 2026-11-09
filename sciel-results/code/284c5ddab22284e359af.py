# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # Safety margin before 630s limit

_N = 13
_TRIANGLE_INDICES = np.array(list(itertools.combinations(range(_N), 3)))

def _min_triangle_area_fast(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Uses precomputed indices for speed.
    """
    p1 = points[_TRIANGLE_INDICES[:, 0]]
    p2 = points[_TRIANGLE_INDICES[:, 1]]
    p3 = points[_TRIANGLE_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area_fast(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _negative_ratio(x):
    """
    Objective function to minimize (negative of the score).
    """
    points = x.reshape(_N, 2)
    return -_score(points)

def _soft_min_ratio(x, beta=10.0):
    """
    Soft minimum of triangle areas using log-sum-exp for smoothness.
    This makes the objective differentiable and easier to optimize.
    """
    points = x.reshape(_N, 2)
    p1 = points[_TRIANGLE_INDICES[:, 0]]
    p2 = points[_TRIANGLE_INDICES[:, 1]]
    p3 = points[_TRIANGLE_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft minimum using log-sum-exp
    log_sum_exp = np.log(np.sum(np.exp(-beta * areas)))
    soft_min = -np.log(np.sum(np.exp(-beta * areas))) / beta
    
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10:
        return 1e10
    
    return -(soft_min / a_hull)

def _lloyd_relax(points, n_iter=50, step_size=0.5):
    """
    Lloyd's algorithm: move each interior point toward the centroid of its
    Voronoi cell. This enforces uniform density, which helps maximize
    the minimum triangle area.
    
    Only moves points that are not on the convex hull.
    """
    points = points.copy()
    
    for _ in range(n_iter):
        if time.time() > _DEADLINE - 60:
            break
        
        try:
            hull = ConvexHull(points)
            hull_indices = set(hull.vertices)
        except Exception:
            break
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except Exception:
            break
        
        for i in range(_N):
            if i in hull_indices:
                continue
            
            # Get Voronoi region for point i
            region = vor.regions[vor.point_region[i]]
            
            if len(region) == 0 or -1 in region:
                continue
            
            # Get vertices of the Voronoi cell
            cell_vertices = np.array([vor.vertices[j] for j in region])
            
            if len(cell_vertices) < 3:
                continue
            
            # Compute centroid of the Voronoi cell
            centroid = np.mean(cell_vertices, axis=0)
            
            # Move point toward centroid
            points[i] = (1 - step_size) * points[i] + step_size * centroid
    
    return points

def _regular_polygon(n, radius=1.0):
    """
    Generate n points on a regular polygon.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_rings(n_outer, n_inner, r_inner=0.5):
    """
    Generate points on two concentric rings.
    """
    outer = _regular_polygon(n_outer, 1.0)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    inner = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    return np.vstack([outer, inner])

def _generate_structured_configs():
    """
    Generate a set of structured initial configurations.
    """
    configs = []
    
    # Regular 13-gon
    configs.append(_regular_polygon(13, 1.0))
    
    # 7 outer + 6 inner
    configs.append(_concentric_rings(7, 6, 0.5))
    configs.append(_concentric_rings(7, 6, 0.4))
    configs.append(_concentric_rings(7, 6, 0.6))
    
    # 6 outer + 7 inner
    configs.append(_concentric_rings(6, 7, 0.5))
    configs.append(_concentric_rings(6, 7, 0.4))
    
    # 5 outer + 8 inner
    configs.append(_concentric_rings(5, 8, 0.5))
    configs.append(_concentric_rings(5, 8, 0.4))
    
    # 4 outer + 9 inner
    configs.append(_concentric_rings(4, 9, 0.5))
    
    # 3 outer + 10 inner
    configs.append(_concentric_rings(3, 10, 0.5))
    
    # 2 outer + 11 inner
    configs.append(_concentric_rings(2, 11, 0.5))
    
    # 1 outer + 12 inner
    configs.append(_concentric_rings(1, 12, 0.5))
    
    return configs

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Uses a multi-phase approach:
    1. Generate structured initial configurations
    2. Apply Lloyd's relaxation to enforce uniform density
    3. Optimize using soft-min surrogate with L-BFGS-B
    4. Polish with exact objective using Powell
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = _N
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Generate and evaluate structured configurations
    configs = _generate_structured_configs()
    
    for config in configs:
        if time.time() > _DEADLINE - 120:
            break
        
        # Apply Lloyd's relaxation
        relaxed = _lloyd_relax(config, n_iter=30, step_size=0.5)
        
        # Evaluate
        s = _score(relaxed)
        if s > best_score:
            best_score = s
            best_points = relaxed.copy()
    
    # Also try random perturbations of the best structured config
    if best_points is not None:
        for trial in range(5):
            if time.time() > _DEADLINE - 120:
                break
            
            perturbed = best_points + rng.random((n, 2)) * 0.1
            relaxed = _lloyd_relax(perturbed, n_iter=30, step_size=0.5)
            s = _score(relaxed)
            if s > best_score:
                best_score = s
                best_points = relaxed.copy()
    
    # Phase 2: Optimize using soft-min surrogate with L-BFGS-B
    if best_points is not None and time.time() < _DEADLINE - 60:
        try:
            res = minimize(
                _soft_min_ratio,
                best_points.flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-14,
                    'gtol': 1e-12
                }
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 3: Polish with exact objective using Powell
    if best_points is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                _negative_ratio,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 5000,
                    'xtol': 1e-12,
                    'ftol': 1e-14,
                    'niter': 200
                }
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 4: Additional local refinement with Nelder-Mead
    if best_points is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                _negative_ratio,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(13, 1.0)
    
    return best_points

# EVOLVE-BLOCK-END
