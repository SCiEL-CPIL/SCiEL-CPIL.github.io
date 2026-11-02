"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
SQRT3 = 3 ** 0.5

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_triangle_area(points, temperature=0.01):
    """Soft-min approximation using log-sum-exp for smoother gradient."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas.append(a)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-a/T)))/T approximates min as T->0
    # But for optimization, we want to maximize min, so minimize -soft_min
    # Use: soft_min = -T * log(sum(exp(-a/T)))
    # To avoid overflow, use log-sum-exp trick
    min_a = np.min(areas)
    shifted = -areas / temperature
    shifted -= np.max(shifted)
    soft_min = -temperature * (np.max(shifted) + np.log(np.sum(np.exp(shifted))))
    return soft_min

def _is_inside_triangle(p):
    """Check if point is inside or on the boundary of the equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # y >= 0
    # y <= sqrt(3) * x  (left edge from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x)
    # y <= sqrt(3) * (1 - x)  (right edge from (1,0) to (0.5, sqrt(3)/2): y = sqrt(3)*(1-x))
    if y < -1e-10:
        return False
    if y > SQRT3 * x + 1e-10:
        return False
    if y > SQRT3 * (1 - x) + 1e-10:
        return False
    return True

def _clip_to_triangle(p):
    """Clip point to be inside the triangle."""
    x, y = p
    # Project to interior if outside
    if y < 0:
        y = 0.001
    if y > SQRT3 * x:
        # Move toward interior
        if x < 0.5:
            x = min(x, 0.5)
            y = min(y, SQRT3 * x - 0.001)
        else:
            x = max(x, 0.5)
            y = min(y, SQRT3 * (1 - x) - 0.001)
    if y > SQRT3 * (1 - x):
        if x > 0.5:
            x = max(x, 0.5)
            y = min(y, SQRT3 * (1 - x) - 0.001)
        else:
            x = min(x, 0.5)
            y = min(y, SQRT3 * x - 0.001)
    x = max(0.001, min(0.999, x))
    y = max(0.001, min(SQRT3 * 0.5 - 0.001, y))
    # Final check
    if not _is_inside_triangle([x, y]):
        # Fallback: use barycentric coordinates
        x = 0.5
        y = SQRT3 * 0.25
    return [x, y]

def _objective(x_flat):
    """Objective to minimize: negative of min triangle area."""
    points = x_flat.reshape(-1, 2)
    # Clip to triangle
    points = np.array([_clip_to_triangle(p) for p in points])
    min_a = _min_triangle_area(points)
    return -min_a

def _objective_soft(x_flat, temperature=0.01):
    """Objective to minimize: negative of soft-min triangle area."""
    points = x_flat.reshape(-1, 2)
    points = np.array([_clip_to_triangle(p) for p in points])
    soft_min = _soft_min_triangle_area(points, temperature)
    return -soft_min

def _random_points_in_triangle(rng, n):
    """Generate n random points inside the equilateral triangle."""
    points = []
    for _ in range(n):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = (SQRT3 / 2) * v
                points.append([x, y])
                break
    return np.array(points)

def _optimize_from_start(points, deadline, max_iter=500):
    """Run local optimization from given starting points."""
    x0 = points.flatten()
    
    # Try L-BFGS-B with the raw min objective first
    best_points = points.copy()
    best_val = _min_triangle_area(best_points)
    
    try:
        res = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxls': 50, 'ftol': 1e-10}
        )
        if time.time() > deadline:
            return best_points, best_val
        
        opt_points = res.x.reshape(-1, 2)
        opt_points = np.array([_clip_to_triangle(p) for p in opt_points])
        opt_val = _min_triangle_area(opt_points)
        
        if opt_val > best_val:
            best_points = opt_points
            best_val = opt_val
    except Exception:
        pass
    
    # Try with soft-min objective for potentially better landscape
    if time.time() < deadline:
        try:
            res2 = minimize(
                lambda x: _objective_soft(x, temperature=0.05),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxls': 50, 'ftol': 1e-10}
            )
            if time.time() > deadline:
                return best_points, best_val
            
            opt_points2 = res2.x.reshape(-1, 2)
            opt_points2 = np.array([_clip_to_triangle(p) for p in opt_points2])
            opt_val2 = _min_triangle_area(opt_points2)
            
            if opt_val2 > best_val:
                best_points = opt_points2
                best_val = opt_val2
        except Exception:
            pass
    
    # Polish with tighter optimization
    if time.time() < deadline:
        try:
            x_polish = best_points.flatten()
            res3 = minimize(
                _objective,
                x_polish,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxls': 50, 'ftol': 1e-12}
            )
            if time.time() > deadline:
                return best_points, best_val
            
            pol_points = res3.x.reshape(-1, 2)
            pol_points = np.array([_clip_to_triangle(p) for p in pol_points])
            pol_val = _min_triangle_area(pol_points)
            
            if pol_val > best_val:
                best_points = pol_points
                best_val = pol_val
        except Exception:
            pass
    
    return best_points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    import random
    rng = random.Random(42)
    
    best_points = None
    best_val = -float('inf')
    
    # Multiple random restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial points
        points = _random_points_in_triangle(rng, N)
        
        # Optimize from this start
        opt_points, opt_val = _optimize_from_start(points, deadline, max_iter=300)
        
        if opt_val > best_val:
            best_val = opt_val
            best_points = opt_points
    
    if best_points is None:
        # Fallback: return random points
        best_points = _random_points_in_triangle(rng, N)
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 5:
        try:
            final_points, final_val = _optimize_from_start(best_points, deadline, max_iter=500)
            if final_val > best_val:
                best_points = final_points
                best_val = final_val
        except Exception:
            pass
    
    return best_points
