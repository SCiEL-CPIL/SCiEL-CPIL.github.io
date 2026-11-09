# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _triangle_areas_vectorized(points):
    """Compute all 286 triangle areas vectorized."""
    n = len(points)
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas


def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(13,3)=286 triangles."""
    return np.min(_triangle_areas_vectorized(points))


def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the Heilbronn score: A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return -1.0
    return a_min / a_hull


def _soft_min_objective(params, temperature=0.01):
    """
    Smooth surrogate for the min triangle area using log-sum-exp.
    Minimizes -soft_min(triangle_areas) / hull_area.
    """
    points = params.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    
    # Soft-min: -log(sum(exp(-areas/T)))/T approximates min(areas) as T->0
    # Use log-sum-exp for numerical stability
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp * temperature
    
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(soft_min) or not np.isfinite(a_hull):
        return 1e10
    
    return -soft_min / a_hull


def _hard_objective(params):
    """Objective to minimize (negative score) for SLSQP."""
    points = params.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    score = _score(points)
    if score < 0:
        return 1e10
    return -score


def _make_seed_hexagonal():
    """Create a hexagonal/triangular lattice seed for 13 points."""
    pts = []
    pts.append([0.0, 0.0])
    pts.append([-0.5, np.sqrt(3)/2])
    pts.append([0.5, np.sqrt(3)/2])
    pts.append([-1.0, np.sqrt(3)])
    pts.append([0.0, np.sqrt(3)])
    pts.append([1.0, np.sqrt(3)])
    pts.append([-1.5, 1.5*np.sqrt(3)])
    pts.append([-0.5, 1.5*np.sqrt(3)])
    pts.append([0.5, 1.5*np.sqrt(3)])
    pts.append([1.5, 1.5*np.sqrt(3)])
    pts.append([-1.0, 2*np.sqrt(3)])
    pts.append([0.0, 2*np.sqrt(3)])
    pts.append([1.0, 2*np.sqrt(3)])
    
    points = np.array(pts)
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _make_seed_grid():
    """Create a grid-based seed for 13 points."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    points = np.array(pts[:13], dtype=float)
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _make_seed_circular():
    """Create a circular seed with center point."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[i + 1] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_mixed():
    """Create a mixed seed: center + inner ring + outer ring."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        points[i + 1] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(8):
        angle = i * 2 * np.pi / 8
        points[i + 5] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_pentagonal():
    """Create a pentagonal arrangement: center + 5 inner + 7 outer."""
    points = np.zeros((13, 2))
    points[0] = [0.0, 0.0]
    # Inner ring: 5 points
    for i in range(5):
        angle = i * 2 * np.pi / 5 - np.pi/2
        points[i + 1] = [0.6 * np.cos(angle), 0.6 * np.sin(angle)]
    # Outer ring: 7 points
    for i in range(7):
        angle = i * 2 * np.pi / 7 - np.pi/2
        points[i + 6] = [np.cos(angle), np.sin(angle)]
    return points


def _make_seed_triangular_ring():
    """Triangular lattice with a different subset for 13 points."""
    # Use a larger triangular grid and pick 13 well-spread points
    pts = []
    for row in range(4):
        for col in range(row + 1):
            x = col - row / 2.0
            y = row * np.sqrt(3) / 2
            pts.append([x, y])
    # We have 1+2+3+4=10 points, need 3 more from row 4
    for col in [0, 1, 2]:
        x = col - 4 / 2.0
        y = 4 * np.sqrt(3) / 2
        pts.append([x, y])
    
    points = np.array(pts[:13])
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def _normalize_points(points):
    """Center and normalize points to unit scale."""
    points = points - points.mean(axis=0)
    scale = np.max(np.abs(points))
    if scale > 0:
        points = points / scale
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses structured seeds with smooth surrogate optimization and final polish.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Generate candidate seeds
    base_seeds = [
        _make_seed_hexagonal(),
        _make_seed_grid(),
        _make_seed_circular(),
        _make_seed_mixed(),
        _make_seed_pentagonal(),
        _make_seed_triangular_ring(),
    ]
    
    # Add perturbed versions
    rng = np.random.default_rng(seed=42)
    seeds = list(base_seeds)
    for seed_pts in base_seeds:
        for _ in range(2):
            perturbed = seed_pts + rng.normal(0, 0.03, (13, 2))
            seeds.append(_normalize_points(perturbed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Use soft-min objective for smooth optimization
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        
        initial_params = _normalize_points(seed).flatten()
        
        try:
            # Use soft-min with decreasing temperature
            for temp in [0.05, 0.02, 0.01]:
                if time.time() > deadline - 30:
                    break
                
                result = minimize(
                    lambda p: _soft_min_objective(p, temperature=temp),
                    initial_params,
                    method='SLSQP',
                    options={'maxiter': 300, 'ftol': 1e-10},
                )
                
                if result.fun < 1e9:
                    opt_points = _normalize_points(result.x.reshape(13, 2))
                    score = _score(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
                        initial_params = opt_points.flatten()
                        
        except Exception:
            continue
    
    # Phase 2: Polish with hard objective (exact min)
    if best_points is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                _hard_objective,
                best_points.flatten(),
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-12},
            )
            
            if result.fun < 1e9:
                polished = _normalize_points(result.x.reshape(13, 2))
                score = _score(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Phase 3: Final ultra-precise polish if time allows
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _hard_objective,
                best_points.flatten(),
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-14},
            )
            
            if result.fun < 1e9:
                polished = _normalize_points(result.x.reshape(13, 2))
                score = _score(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _normalize_points(_make_seed_hexagonal())
    
    # Final verification
    if not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        best_points = _normalize_points(_make_seed_hexagonal())
    
    return best_points


# EVOLVE-BLOCK-END
