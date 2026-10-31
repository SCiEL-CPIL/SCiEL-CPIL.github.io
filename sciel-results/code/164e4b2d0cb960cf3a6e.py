"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y < -tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _get_10pt_config():
    """Return a known good 10-point configuration for Heilbronn problem."""
    # Use a hexagonal-like arrangement that's known to work well
    h = np.sqrt(3) / 2
    # 10 points: corners + edge midpoints + interior
    # Scaled to fit in unit triangle
    points = []
    # 3 corners
    points.append([0.05, 0.05])
    points.append([0.95, 0.05])
    points.append([0.5, h - 0.05])
    # 3 edge midpoints (slightly inward)
    points.append([0.5, 0.08])
    points.append([0.08, 0.08 + 2*h*0.08])  # left edge
    points.append([0.92, 0.08 + 2*h*0.08])  # right edge
    # 4 interior points in a diamond pattern
    cx, cy = 0.5, h/3
    r = 0.15
    points.append([cx - r*0.7, cy])
    points.append([cx + r*0.7, cy])
    points.append([cx, cy + r*0.7])
    points.append([cx, cy - r*0.7])
    return np.array(points)

def _build_cluster_config(params):
    """
    Build 11 points from rigid cluster parameters.
    params: [scale, rotation, tx, ty, px, py]
    First 10 points: rigid transform of base config
    Last point: free point (px, py)
    """
    scale, rotation, tx, ty, px, py = params
    base = _get_10pt_config()
    
    # Rotate
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    # Scale, rotate, translate
    cluster = (base - 0.5) * scale @ R.T + np.array([tx, ty])
    
    # Add free point
    points = np.vstack([cluster, [[px, py]]])
    return points

def _cluster_objective(params):
    """Objective: negative min triangle area with penalty for out-of-bounds."""
    points = _build_cluster_config(params)
    
    # Check bounds
    h = np.sqrt(3) / 2
    penalty = 0.0
    for p in points:
        x, y = p[0], p[1]
        if x < -0.01 or x > 1.01 or y < -0.01 or y > h + 0.01:
            penalty += 1.0
        if y > 2*h*x + 0.01:
            penalty += 1.0
        if y > 2*h*(1-x) + 0.01:
            penalty += 1.0
    
    if penalty > 0:
        return -0.001 + penalty * 10
    
    min_area = _min_triangle_area(points)
    return -min_area

def _optimize_rigid_cluster(deadline):
    """Optimize the rigid cluster + free point configuration."""
    h = np.sqrt(3) / 2
    
    # Bounds: scale, rotation, tx, ty, px, py
    bounds = [
        (0.5, 1.2),   # scale
        (-0.5, 0.5),  # rotation
        (0.1, 0.9),   # tx
        (0.05, h-0.05),  # ty
        (0.02, 0.98),    # px
        (0.02, h-0.02)   # py
    ]
    
    best_params = None
    best_obj = float('inf')
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            _cluster_objective, bounds,
            seed=42, maxiter=200, popsize=20,
            tol=1e-10, mutation=(0.5, 1.5), recombination=0.8,
            polish=True, init='sobol',
            update_interval=1
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass
    
    # Try multiple random starts with local optimization
    rng = np.random.RandomState(123)
    for trial in range(15):
        if time.time() > deadline - 20:
            break
        
        x0 = np.array([
            rng.uniform(0.6, 1.1),
            rng.uniform(-0.3, 0.3),
            rng.uniform(0.2, 0.8),
            rng.uniform(0.1, h-0.1),
            rng.uniform(0.1, 0.9),
            rng.uniform(0.1, h-0.1)
        ])
        
        try:
            result = minimize(_cluster_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        points = _build_cluster_config(best_params)
        if _all_inside(points):
            return points, _min_triangle_area(points)
    
    return None, 0.0

def _local_polish(points, deadline):
    """Local refinement using Nelder-Mead on flattened parameters."""
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    # Use multiple local refinement passes
    for pass_num in range(4):
        if time.time() > deadline - 5:
            break
        
        def objective(params):
            pts = params.reshape(-1, 2)
            if not _all_inside(pts):
                return 10.0
            return -_min_triangle_area(pts)
        
        x0 = points.flatten()
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _all_inside(new_points):
                    new_min_area = _min_triangle_area(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
        
        # Coordinate descent refinement
        if time.time() > deadline - 3:
            break
        
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                for coord in range(2):
                    for step in [0.005, 0.002, 0.001, 0.0005, 0.0002]:
                        old_val = points[i, coord]
                        points[i, coord] = old_val + step
                        if _all_inside(points):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val - step
                        if _all_inside(points):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val
            if not improved:
                break
    
    return points, current_min_area

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Rigid cluster optimization (primary)
    if time.time() < deadline - 30:
        cluster_points, cluster_area = _optimize_rigid_cluster(deadline)
        if cluster_points is not None:
            if cluster_area > best_min_area:
                best_min_area = cluster_area
                best_points = cluster_points
    
    # Strategy 2: Full differential evolution as backup
    if best_points is None or time.time() < deadline - 40:
        def _negative_min_area(params):
            points = params.reshape(-1, 2)
            if not _all_inside(points):
                penalty = 0.0
                for p in points:
                    x, y = p[0], p[1]
                    if x < 0 or x > 1 or y < 0 or y > h:
                        penalty += 10.0
                    if y > 2*h*x + 1e-4:
                        penalty += 10.0
                    if y > 2*h*(1-x) + 1e-4:
                        penalty += 10.0
                return -0.001 + penalty
            return -_min_triangle_area(points)
        
        bounds = [(0, 1)] * 2 * N
        try:
            de_result = differential_evolution(
                _negative_min_area, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _all_inside(de_points):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Random initializations with local polish
    if best_points is None or time.time() < deadline - 20:
        rng = np.random.RandomState(123)
        
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            
            points = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 0.95 and u > 0.02 and v > 0.02:
                        px = u + 0.5 * v
                        py = h * v
                        points.append([px, py])
                        break
            
            points = np.array(points)
            polished, min_area = _local_polish(points, deadline)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    if best_points is None:
        # Fallback: simple grid
        best_points = np.array([[0.5, h/3]] + [[0.2+i*0.15, 0.1+j*0.1] for i in range(3) for j in range(3)][:10])
    
    return best_points
# EVOLVE-BLOCK-END
