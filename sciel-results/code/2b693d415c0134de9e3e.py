import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(points):
    """
    Objective to maximize: min_area - penalty
    """
    area = _min_area(points)
    pen = _penalty(points)
    return area - 1000.0 * pen

def _smooth_objective(points, beta=50.0):
    """
    Smooth approximation of min area using log-sum-exp.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    min_approx = -np.log(np.sum(np.exp(-beta * areas))) / beta
    
    pen = _penalty(points)
    return min_approx - 1000.0 * pen

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population(rng, count=1):
    """
    Generate initial valid configurations.
    """
    configs = []
    for _ in range(count):
        pts = []
        for _ in range(N):
            pts.append(_random_valid_point(rng))
        configs.append(np.array(pts))
    return configs

def _simulated_annealing(initial_points, time_limit, rng):
    """
    Run Simulated Annealing with adaptive step size.
    """
    current_points = initial_points.copy()
    current_score = _objective(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.05
    cooling_rate = 0.9995
    min_T = 1e-6
    step_scale = 0.05
    
    start_time = time.time()
    no_improve_count = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move
        idx = rng.integers(N)
        noise = rng.normal(0, step_scale, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += noise
        
        new_score = _objective(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > 100:
            step_scale = max(0.001, step_scale * 0.9)
            no_improve_count = 0
        elif no_improve_count == 0:
            step_scale = min(0.1, step_scale * 1.05)
        
        T *= cooling_rate
        
        # Occasionally, do a larger jump
        if rng.random() < 0.005:
             idx = rng.integers(N)
             new_points = current_points.copy()
             new_points[idx] = _random_valid_point(rng)
             new_score = _objective(new_points)
             delta = new_score - current_score
             if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()
    
    return best_points, best_score

def _delaunay_refine(points, time_limit, rng):
    """
    Refine points using Delaunay triangulation structure.
    This is a structural change: we use the Delaunay triangulation to identify
    the critical triangles and optimize specifically for them.
    """
    current_points = points.copy()
    current_score = _objective(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.02
    cooling_rate = 0.999
    min_T = 1e-7
    step_scale = 0.02
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(current_points)
            # Get the triangles from Delaunay
            tri_verts = tri.simplices  # shape (M, 3)
            
            # Compute areas of Delaunay triangles
            Pi = current_points[tri_verts[:, 0]]
            Pj = current_points[tri_verts[:, 1]]
            Pk = current_points[tri_verts[:, 2]]
            
            v1 = Pj - Pi
            v2 = Pk - Pi
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            tri_areas = 0.5 * np.abs(cross)
            
            # Identify the smallest Delaunay triangle
            min_idx = np.argmin(tri_areas)
            min_tri = tri_verts[min_idx]
            min_area_delaunay = tri_areas[min_idx]
            
            # Focus optimization on points in the smallest triangle
            # Move the points in the smallest triangle to increase its area
            for pt_idx in min_tri:
                if time.time() - start_time > time_limit:
                    break
                    
                # Try moving this point
                noise = rng.normal(0, step_scale, size=2)
                new_points = current_points.copy()
                new_points[pt_idx] += noise
                
                new_score = _objective(new_points)
                delta = new_score - current_score
                
                if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                    current_points = new_points
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
        except Exception:
            # Delaunay failed (likely due to collinear points or other issues)
            # Fall back to standard SA move
            idx = rng.integers(N)
            noise = rng.normal(0, step_scale, size=2)
            new_points = current_points.copy()
            new_points[idx] += noise
            new_score = _objective(new_points)
            delta = new_score - current_score
            
            if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        
        T *= cooling_rate
        step_scale = max(0.001, step_scale * 0.999)
    
    return best_points, best_score

def _grid_search_initialization():
    """
    Generate a structured initial configuration using a grid-like pattern
    within the triangle. This provides a better starting point than pure random.
    """
    points = []
    
    # Add the three vertices
    points.append(np.array([0.0, 0.0]))
    points.append(np.array([1.0, 0.0]))
    points.append(np.array([0.5, H]))
    
    # Add midpoints of edges
    points.append(np.array([0.5, 0.0]))
    points.append(np.array([0.25, H/2]))
    points.append(np.array([0.75, H/2]))
    
    # Add center point
    points.append(np.array([0.5, H/3]))
    
    # Add a few more points to reach 11
    # Use a hexagonal-like pattern
    remaining = 11 - len(points)
    for i in range(remaining):
        # Distribute remaining points roughly evenly
        angle = 2 * np.pi * i / remaining
        r = 0.15
        cx = 0.5 + r * np.cos(angle)
        cy = H/3 + r * np.sin(angle) * (H/0.5)  # Scale y to fit triangle aspect
        points.append(np.array([cx, cy]))
    
    points = np.array(points)
    
    # Ensure all points are inside the triangle
    for i in range(len(points)):
        x, y = points[i]
        # Clip to triangle
        if y < 0:
            y = 0
        if y > S3 * x:
            x = y / S3
        if y > S3 * (1 - x):
            x = 1 - y / S3
        points[i] = np.array([x, y])
    
    return points

def construct():
    start_time = time.time()
    timeout = 120.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multi-start SA with structured initialization
    num_starts = 4
    time_per_start = (timeout - 40) / num_starts  # Leave 40s for Delaunay refinement + polishing
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 40:
            break
        
        # Use structured initialization for first start, random for others
        if i == 0:
            init_pts = _grid_search_initialization()
        else:
            init_pts = _initial_population(rng, 1)[0]
        
        # Run SA
        sa_pts, sa_score = _simulated_annealing(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Phase 2: Delaunay-based refinement (structural change)
    if best_points is not None and time.time() - start_time < timeout - 20:
        delaunay_time = timeout - 20 - (time.time() - start_time)
        delaunay_pts, delaunay_score = _delaunay_refine(best_points, delaunay_time, rng)
        
        if delaunay_score > best_score:
            best_score = delaunay_score
            best_points = delaunay_pts
    
    # Phase 3: Polish with L-BFGS-B using smooth objective
    if best_points is not None and time.time() - start_time < timeout - 10:
        def _min_smooth_obj(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            return -_smooth_objective(pts)
        
        try:
            res = minimize(
                _min_smooth_obj, 
                best_points.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            polished_pts = res.x.reshape(-1, 2)
            polished_score = _objective(polished_pts)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
        
        # Second polish with tighter tolerance if time remains
        if time.time() - start_time < timeout - 5:
            try:
                res2 = minimize(
                    _min_smooth_obj, 
                    best_points.flatten(), 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                polished_pts2 = res2.reshape(-1, 2)
                polished_score2 = _objective(polished_pts2)
                
                if polished_score2 > best_score:
                    best_score = polished_score2
                    best_points = polished_pts2
            except Exception:
                pass
    
    # Final validity check: clip points to triangle if slightly outside
    if best_points is None:
        best_points = _grid_search_initialization()
    
    x = best_points[:, 0]
    y = best_points[:, 1]
    
    # Clip y >= 0
    y = np.maximum(y, 0)
    
    # Clip y <= sqrt(3)*x
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    
    # Clip y <= sqrt(3)*(1-x)
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
    
    # Re-check after clipping
    y = np.maximum(y, 0)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
    
    best_points[:, 0] = x
    best_points[:, 1] = y
    
    return best_points
