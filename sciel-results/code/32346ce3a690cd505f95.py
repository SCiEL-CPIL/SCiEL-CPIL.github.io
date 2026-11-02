import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 150.0
SAFETY_MARGIN = 10.0
START_TIME = time.time()

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation for all triples
    # N=13 -> 286 triangles, so memory is trivial
    # We can use broadcasting to compute all areas at once
    
    # indices
    i, j, k = np.indices((n, n, n))
    # Only consider i < j < k
    mask = (i < j) & (j < k)
    
    # Extract points
    pi = points[i[mask]]
    pj = points[j[mask]]
    pk = points[k[mask]]
    
    # Vectors
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product (z-component)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    # Areas
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """
    Objective function to minimize.
    We want to maximize min_triangle_area / hull_area.
    So we minimize - (min_triangle_area / hull_area).
    """
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 1e10  # Large penalty for degenerate hull
    
    ratio = min_area / hull_area
    return -ratio

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def _optimize_from_start(start_points, max_iter=500, tol=1e-8):
    """
    Run Nelder-Mead optimization from a given start point.
    Returns the optimized points and score.
    """
    x0 = start_points.flatten()
    
    # Use Nelder-Mead for robustness with non-smooth functions
    res = minimize(
        _score,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iter,
            'xatol': tol,
            'fatol': tol
        }
    )
    
    if res.fun < _score(x0):
        pts = res.x.reshape(-1, 2)
        score = -res.fun
        return pts, score
    else:
        return start_points, -_score(start_points)

def _make_initial_configurations():
    """
    Generate a diverse set of initial configurations.
    """
    inits = []
    
    # 1. Regular polygon (baseline)
    inits.append(_regular_polygon(N, radius=1.0))
    
    # 2. Regular polygon with various angle offsets
    for offset in np.linspace(0, 2 * np.pi / N, 5, endpoint=False):
        inits.append(_regular_polygon(N, radius=1.0, angle_offset=offset))
    
    # 3. Jittered regular polygons
    for _ in range(15):
        pts = _regular_polygon(N, radius=1.0)
        noise = np.random.normal(0, 0.15, pts.shape)
        pts += noise
        inits.append(pts)
    
    # 4. Random points in unit disk
    for _ in range(15):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0.2, 1.0, N))  # Avoid center to spread points
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))
    
    # 5. Concentric rings
    # Inner ring: 5 points, Outer ring: 8 points
    for _ in range(5):
        inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.random.uniform(0, 2 * np.pi / 5)
        outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.random.uniform(0, 2 * np.pi / 8)
        
        inner_r = 0.4 + np.random.uniform(-0.05, 0.05)
        outer_r = 1.0 + np.random.uniform(-0.05, 0.05)
        
        inner_pts = np.column_stack((inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)))
        outer_pts = np.column_stack((outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)))
        
        pts = np.vstack((inner_pts, outer_pts))
        inits.append(pts)
    
    # 6. Grid-like points with perturbation
    # 4x4 grid = 16 points, remove 3 to get 13
    for _ in range(5):
        grid = np.array([[i * 0.5 - 0.75, j * 0.5 - 0.75] for i in range(4) for j in range(4)])
        # Remove 3 random points
        indices_to_remove = np.random.choice(16, 3, replace=False)
        mask = np.ones(16, dtype=bool)
        mask[indices_to_remove] = False
        pts = grid[mask]
        # Normalize to unit area approx
        pts = pts - np.mean(pts, axis=0)
        scale = 1.0 / np.sqrt(np.max(np.abs(pts)))
        pts = pts * scale
        # Add small noise
        pts += np.random.normal(0, 0.05, pts.shape)
        inits.append(pts)
    
    return inits

def construct():
    global START_TIME
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    # Generate initial configurations
    inits = _make_initial_configurations()
    
    # Optimization Loop
    for init in inits:
        # Check time
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            # Run optimization
            pts, score = _optimize_from_start(init, max_iter=300, tol=1e-7)
            
            if score > best_score:
                best_score = score
                best_points = pts
                
        except Exception:
            continue
    
    # If we have time, try Powell method on the best point found so far
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        try:
            x0 = best_points.flatten()
            res = minimize(
                _score,
                x0,
                method='Powell',
                options={
                    'maxiter': 200,
                    'xtol': 1e-8,
                    'ftol': 1e-8
                }
            )
            
            if res.fun < _score(x0):
                pts = res.x.reshape(-1, 2)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
                    
        except Exception:
            pass
    
    # If no optimization succeeded, fallback to regular polygon
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
        best_score = -_score(best_points)
        
    return best_points

# EVOLVE-BLOCK-START
# (The above is the complete program; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
