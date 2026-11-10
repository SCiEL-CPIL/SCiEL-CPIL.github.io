# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area over all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    # Compute areas for all triples using vectorized cross product
    # Area of triangle (p_i, p_j, p_k) = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    min_area = np.inf
    # Use a chunked approach to avoid memory blowup
    # For 13 points, there are 286 triples, so we can do it all at once
    idx_i = np.repeat(np.arange(n), n*(n-1)//2)
    idx_j = np.tile(np.repeat(np.arange(n), n-1), n//1)  # this is tricky, let's do it differently
    
    # Simpler: generate all triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    triples = np.array(triples)
    pi = points[triples[:, 0]]
    pj = points[triples[:, 1]]
    pk = points[triples[:, 2]]
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _hull_area(points):
    """Compute convex hull area, return 0 on failure."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute A_min / A_hull."""
    a_min = _min_triangle_area_vec(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
    return a_min / a_hull


def _neg_log_min_area(points):
    """Negative log of min triangle area for minimization."""
    a_min = _min_triangle_area_vec(points)
    if a_min <= 1e-15:
        return 1e15
    return -np.log(a_min)


def _objective(points):
    """Objective to minimize: negative log of (A_min / A_hull)."""
    a_min = _min_triangle_area_vec(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_min <= 1e-15:
        return 1e15
    return -np.log(a_min / a_hull)


def _make_initial_configs():
    """Generate diverse initial configurations for 13 points."""
    configs = []
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 13-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(np.vstack([pts, [[0, 0]]]))
    
    # 3. Two concentric rings: 7 outer, 6 inner
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = np.column_stack([0.5*np.cos(angles_inner), 0.5*np.sin(angles_inner)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. Three concentric rings: 5, 4, 4
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    angles3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/8
    r1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    r2 = np.column_stack([0.6*np.cos(angles2), 0.6*np.sin(angles2)])
    r3 = np.column_stack([0.3*np.cos(angles3), 0.3*np.sin(angles3)])
    configs.append(np.vstack([r1, r2, r3]))
    
    # 5. Regular 13-gon slightly perturbed
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    radii = 1.0 + 0.1 * np.sin(3*angles)
    configs.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]))
    
    # 6. Square-like grid arrangement
    # 4x4 grid minus one corner, scaled
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])[:13]
    configs.append(pts)
    
    # 7. Hexagonal lattice portion
    pts_hex = []
    for row in range(4):
        for col in range(4):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3)/2
            pts_hex.append([x, y])
            if len(pts_hex) >= 13:
                break
        if len(pts_hex) >= 13:
            break
    configs.append(np.array(pts_hex[:13]))
    
    # 8. Random with good seed
    rng = np.random.default_rng(123)
    configs.append(rng.random((13, 2)))
    
    # 9. Another random
    rng = np.random.default_rng(456)
    configs.append(rng.random((13, 2)))
    
    # 10. Regular 13-gon with varying radii
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    radii = np.array([1.0, 0.8, 0.9, 0.7, 1.0, 0.85, 0.95, 0.75, 1.0, 0.9, 0.8, 0.95, 0.85])
    configs.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]))
    
    return configs


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the
    smallest triangle area to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock, with safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Evaluate initial configs
    configs = _make_initial_configs()
    
    for cfg in configs:
        if time.time() > deadline - 30:
            break
        score = _score(cfg)
        if score > best_score:
            best_score = score
            best_points = cfg.copy()
    
    # Refine using optimization
    # Use L-BFGS-B to minimize negative log of the ratio
    for cfg in configs:
        if time.time() > deadline - 60:
            break
        
        # Try optimization from this starting point
        try:
            result = minimize(
                _objective,
                cfg,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxls': 50}
            )
            if result.success or result.fun < _objective(cfg):
                refined = result.x
                score = _score(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined.copy()
        except Exception:
            pass
    
    # Additional local refinement around the best point
    if best_points is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                _objective,
                best_points,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 100}
            )
            if result.fun < _objective(best_points):
                refined = result.x
                score = _score(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined.copy()
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Validate: check for NaN/inf
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure shape is (13, 2)
    best_points = best_points.reshape(13, 2)
    
    return best_points


# EVOLVE-BLOCK-END
