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

def _negative_min_area(params):
    """Objective for differential evolution: negative of min triangle area."""
    points = params.reshape(-1, 2)
    if not _all_inside(points):
        penalty = 0.0
        for p in points:
            h = np.sqrt(3) / 2
            x, y = p[0], p[1]
            if x < 0 or x > 1 or y < 0 or y > h:
                penalty += 10.0
            if y > 2*h*x + 1e-4:
                penalty += 10.0
            if y > 2*h*(1-x) + 1e-4:
                penalty += 10.0
        return -0.001 + penalty
    return -_min_triangle_area(points)

def _local_polish(points, deadline):
    """Local refinement using Nelder-Mead and coordinate descent."""
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        
        def objective(params):
            pts = params.reshape(-1, 2)
            if not _all_inside(pts):
                return 10.0
            return -_min_triangle_area(pts)
        
        try:
            result = minimize(objective, points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _all_inside(new_points):
                    new_min_area = _min_triangle_area(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
        
        if time.time() > deadline - 3:
            break
        
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                for coord in range(2):
                    for step in [0.01, 0.005, 0.001, 0.0005]:
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

def _build_structured_config():
    """Build a structured 11-point configuration based on known Heilbronn-type patterns."""
    h = np.sqrt(3) / 2
    points = []
    
    eps = 0.02
    points.append([eps, eps])
    points.append([1-eps, eps])
    points.append([0.5, h - eps*2])
    
    points.append([0.5, eps])
    points.append([1-eps, h/2])
    points.append([eps, h/2])
    
    points.append([0.5, h/3])
    
    cx, cy = 0.5, h/3
    r = 0.15
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi])
    for a in angles:
        px = cx + r*np.cos(a)
        py = cy + r*np.sin(a)*0.9
        if _inside_triangle([px, py], tol=1e-3):
            points.append([px, py])
    
    return np.array(points[:N])

def _nlp_solve(points_init, deadline):
    """
    NLP feasibility approach: maximize alpha subject to all triangle areas >= alpha.
    Uses SLSQP with constraints.
    """
    if time.time() > deadline - 10:
        return points_init, _min_triangle_area(points_init)
    
    # Precompute all triangle index combinations
    tri_indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_indices.append((i, j, k))
    
    def _area_constraint(params, idx_tuple):
        i, j, k = idx_tuple
        points = params.reshape(-1, 2)
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        return area
    
    # We'll use a simpler approach: maximize min area via SLSQP
    # by treating it as: minimize -min_area subject to points inside triangle
    
    def objective(params):
        points = params.reshape(-1, 2)
        if not _all_inside(points, tol=1e-4):
            return 100.0
        return -_min_triangle_area(points)
    
    # Define bounds for each coordinate
    h = np.sqrt(3) / 2
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, h))    # y
    
    # Define inequality constraints for triangle boundaries
    def _constraint_left(params):
        """y <= 2*h*x (left edge)"""
        points = params.reshape(-1, 2)
        return np.array([2*h*points[i,0] - points[i,1] for i in range(N)])
    
    def _constraint_right(params):
        """y <= 2*h*(1-x) (right edge)"""
        points = params.reshape(-1, 2)
        return np.array([2*h*(1-points[i,0]) - points[i,1] for i in range(N)])
    
    constraints = [
        {'type': 'ineq', 'fun': _constraint_left},
        {'type': 'ineq', 'fun': _constraint_right}
    ]
    
    try:
        result = minimize(objective, points_init.flatten(), method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-10})
        
        if result.fun < 100.0:
            new_points = result.x.reshape(-1, 2)
            if _all_inside(new_points, tol=1e-3):
                new_min_area = _min_triangle_area(new_points)
                return new_points, new_min_area
    except Exception:
        pass
    
    return points_init, _min_triangle_area(points_init)

def construct():
    deadline = time.time() + 135
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Differential evolution for global search
    if time.time() < deadline - 40:
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
    
    # Strategy 2: Multi-start with local polish
    if best_points is None or time.time() < deadline - 30:
        rng = np.random.RandomState(123)
        
        for seed in range(8):
            if time.time() > deadline - 20:
                break
            
            points = []
            for _ in range(N):
                for _attempt in range(100):
                    u, v = rng.random(), rng.random()
                    if u + v < 0.95 and u > 0.02 and v > 0.02:
                        px = u + 0.5 * v
                        py = h * v
                        points.append([px, py])
                        break
            
            if len(points) < N:
                continue
            
            points = np.array(points)
            polished, min_area = _local_polish(points, deadline)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Strategy 3: NLP approach from structured and best initializations
    if time.time() < deadline - 15:
        # Try NLP from structured config
        structured = _build_structured_config()
        nlp_points, nlp_area = _nlp_solve(structured, deadline)
        if nlp_area > best_min_area:
            best_min_area = nlp_area
            best_points = nlp_points.copy()
        
        # Try NLP from best found so far
        if best_points is not None and time.time() < deadline - 10:
            nlp_points, nlp_area = _nlp_solve(best_points, deadline)
            if nlp_area > best_min_area:
                best_min_area = nlp_area
                best_points = nlp_points.copy()
    
    # Strategy 4: Final polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
