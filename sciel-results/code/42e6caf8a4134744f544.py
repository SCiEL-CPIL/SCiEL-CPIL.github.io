"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay
import itertools

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

def _negative_min_area(params):
    """Objective for differential evolution: negative of min triangle area."""
    points = params.reshape(-1, 2)
    if not _all_inside(points):
        penalty = 0.0
        h = np.sqrt(3) / 2
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

def _soft_min_area(points, beta=50.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Lower beta = smoother (more like geometric mean), higher beta = closer to true min.
    """
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas[i,j,k] = area
                areas[k,j,i] = area  # symmetric fill for completeness
    # Use only unique triangles
    tri_areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_areas.append(areas[i,j,k])
    tri_areas = np.array(tri_areas)
    # Soft min: -log(sum(exp(-beta * a))) / beta
    # Numerically stable version
    min_a = np.min(tri_areas)
    exp_vals = np.exp(-beta * (tri_areas - min_a))
    soft_min = min_a - np.log(np.sum(exp_vals)) / beta
    return soft_min

def _local_polish(points, deadline):
    """Local refinement using Powell's method (robust derivative-free) on the hard objective."""
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    # Powell's method for robust derivative-free optimization
    def objective(params):
        pts = params.reshape(-1, 2)
        if not _all_inside(pts):
            h = np.sqrt(3) / 2
            penalty = 0.0
            for p in pts:
                x, y = p[0], p[1]
                if x < -1e-3 or x > 1+1e-3 or y < -1e-3 or y > h+1e-3:
                    penalty += 5.0
                if y > 2*h*x + 1e-3:
                    penalty += 5.0
                if y > 2*h*(1-x) + 1e-3:
                    penalty += 5.0
            return 1.0 + penalty
        return -_min_triangle_area(pts)
    
    for pass_num in range(4):
        if time.time() > deadline - 8:
            break
        
        x0 = points.flatten()
        
        try:
            result = minimize(objective, x0, method='Powell',
                             options={'maxiter': 300, 'xtol': 1e-9, 'ftol': 1e-11, 'maxfev': 5000})
            if result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _all_inside(new_points):
                    new_min_area = _min_triangle_area(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
        
        # Fine-tuning with coordinate descent
        for _ in range(50):
            if time.time() > deadline - 4:
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

def _delaunay_guided_polish(points, deadline):
    """
    Use Delaunay triangulation to identify critical triangles and refine them.
    The smallest Delaunay triangles are likely to contain the global minimum area triangle.
    """
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    for iteration in range(20):
        if time.time() > deadline - 10:
            break
        
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(points)
            
            # Find the smallest triangles in the Delaunay triangulation
            tri_areas = []
            for simplex in tri.simplices:
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                tri_areas.append((area, simplex))
            
            tri_areas.sort(key=lambda x: x[0])
            
            # Focus on the 3 smallest triangles
            critical_triangles = [t[1] for t in tri_areas[:3]]
            
            # Try to improve by moving vertices of critical triangles
            for simplex in critical_triangles:
                for idx in simplex:
                    # Try small perturbations of this vertex
                    for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                        for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                            if dx == 0 and dy == 0:
                                continue
                            old_x, old_y = points[idx]
                            points[idx] = [old_x + dx, old_y + dy]
                            if _all_inside(points):
                                new_area = _min_triangle_area(points)
                                if new_area > current_min_area:
                                    current_min_area = new_area
                                else:
                                    points[idx] = [old_x, old_y]
                            else:
                                points[idx] = [old_x, old_y]
        except Exception:
            break
    
    return points, current_min_area

def _build_hexagonal_config():
    """
    Build a 10-point hexagonal-like cluster configuration centered at origin.
    """
    dx = 0.25
    dy = dx * np.sqrt(3) / 2
    
    points = []
    # Row 0: y = 1.5*dy, x = -dx/2, dx/2
    points.append([-dx/2, 1.5*dy])
    points.append([dx/2, 1.5*dy])
    # Row 1: y = 0.5*dy, x = -dx, 0, dx
    points.append([-dx, 0.5*dy])
    points.append([0, 0.5*dy])
    points.append([dx, 0.5*dy])
    # Row 2: y = -0.5*dy, x = -dx, 0, dx
    points.append([-dx, -0.5*dy])
    points.append([0, -0.5*dy])
    points.append([dx, -0.5*dy])
    # Row 3: y = -1.5*dy, x = -dx/2, dx/2
    points.append([-dx/2, -1.5*dy])
    points.append([dx/2, -1.5*dy])
    
    return np.array(points)

def _rigid_cluster_objective(params):
    """
    Objective for optimizing rigid cluster placement.
    params: [tx, ty, angle, scale, fx, fy]
    """
    tx, ty, angle, scale, fx, fy = params
    
    base_cluster = _build_hexagonal_config()
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    cluster_pts = (base_cluster * scale) @ rot_mat.T + np.array([tx, ty])
    
    all_points = np.vstack([cluster_pts, [[fx, fy]]])
    
    if not _all_inside(all_points, tol=1e-4):
        h = np.sqrt(3) / 2
        penalty = 0.0
        for p in all_points:
            x, y = p[0], p[1]
            if x < -1e-3 or x > 1+1e-3 or y < -1e-3 or y > h+1e-3:
                penalty += 5.0
            if y > 2*h*x + 1e-3:
                penalty += 5.0
            if y > 2*h*(1-x) + 1e-3:
                penalty += 5.0
        return 1.0 + penalty
    
    min_area = _min_triangle_area(all_points)
    return -min_area

def _optimize_rigid_cluster(deadline):
    """
    Optimize the rigid cluster placement using differential evolution in 6D.
    """
    h = np.sqrt(3) / 2
    
    bounds = [
        (0.1, 0.9),
        (0.1, h-0.1),
        (-np.pi/6, np.pi/6),
        (0.3, 0.8),
        (0.0, 1.0),
        (0.0, h),
    ]
    
    try:
        de_result = differential_evolution(
            _rigid_cluster_objective, bounds,
            seed=42, maxiter=40, popsize=10,
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.8,
            polish=False, init='sobol'
        )
        params = de_result.x
        tx, ty, angle, scale, fx, fy = params
        
        base_cluster = _build_hexagonal_config()
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        cluster_pts = (base_cluster * scale) @ rot_mat.T + np.array([tx, ty])
        all_points = np.vstack([cluster_pts, [[fx, fy]]])
        
        if _all_inside(all_points):
            return all_points
    except Exception:
        pass
    
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Rigid cluster optimization (reduces search dimension)
    if time.time() < deadline - 60:
        cluster_points = _optimize_rigid_cluster(deadline)
        if cluster_points is not None:
            polished, min_area = _local_polish(cluster_points, deadline)
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Strategy 2: Differential evolution for global search
    if time.time() < deadline - 40:
        bounds = [(0, 1)] * 2 * N
        
        try:
            de_result = differential_evolution(
                _negative_min_area, bounds,
                seed=42, maxiter=60, popsize=12,
                tol=1e-7, mutation=(0.5, 1.5), recombination=0.7,
                polish=False, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _all_inside(de_points):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Multiple random initializations with local polish
    if best_points is None or time.time() < deadline - 25:
        rng = np.random.RandomState(123)
        
        for seed in range(6):
            if time.time() > deadline - 20:
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
    
    # Strategy 4: Delaunay-guided refinement on best result
    if best_points is not None and time.time() < deadline - 15:
        best_points, best_min_area = _delaunay_guided_polish(best_points, deadline)
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
