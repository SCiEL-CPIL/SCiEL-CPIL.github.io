# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    h = np.sqrt(3) / 2
    # Barycentric coordinates check
    # p = u*A + v*B + w*C, u+v+w=1, u,v,w >= 0
    # A=(0,0), B=(1,0), C=(0.5, h)
    # p = (x, y)
    # y = v*0 + w*h => w = y/h
    # x = u*0 + v*1 + w*0.5 => v = x - 0.5*w
    # u = 1 - v - w
    w = p[1] / h
    v = p[0] - 0.5 * w
    u = 1 - v - w
    return u >= -tol and v >= -tol and w >= -tol

def _is_valid(points):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p) for p in points)

def _objective(points_flat):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = points_flat.reshape(-1, 2)
    if not _is_valid(points):
        # Soft penalty
        penalty = 0.0
        h = np.sqrt(3) / 2
        for p in points:
            # Distances to boundaries
            # y >= 0
            if p[1] < 0: penalty += p[1] * p[1]
            # x <= 1
            if p[0] > 1: penalty += (p[0] - 1) * (p[0] - 1)
            # x >= 0.5 * (1 - y/h) ? No, x >= (y/h) * 0.5 ?
            # Line AC: y = -sqrt(3) x + sqrt(3). x + y/sqrt(3) <= 1 ?
            # Line BC: y = sqrt(3) (x - 1). y <= sqrt(3)(x-1) -> y - sqrt(3)x + sqrt(3) <= 0
            # Let's use the barycentric check for penalty
            w = p[1] / h
            v = p[0] - 0.5 * w
            u = 1 - v - w
            if u < 0: penalty += u * u
            if v < 0: penalty += v * v
            if w < 0: penalty += w * w
        return 1e5 + penalty
        
    min_area = _min_triangle_area(points)
    return -min_area

def _hex_initial_guess():
    """Generate initial points from a hexagonal lattice pattern."""
    h = np.sqrt(3) / 2
    pts = []
    for j in range(4):
        y = j * h / 3.0
        x_min = 0.5 * j / 3.0
        x_max = 1.0 - 0.5 * j / 3.0
        if x_min > x_max:
            continue
        n_pts = 4 - j
        if n_pts <= 0: continue
        for i in range(n_pts):
            if n_pts == 1:
                x = 0.5
            else:
                x = x_min + (x_max - x_min) * i / (n_pts - 1)
            pts.append([x, y])
            
    import random
    rng = random.Random(42)
    for _ in range(100):
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = [u + 0.5*v, h*v]
            min_dist = min(np.linalg.norm(np.array(p) - np.array(q)) for q in pts)
            if min_dist > 0.05:
                pts.append(p)
                break
                
    return np.array(pts).flatten()

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    h = np.sqrt(3) / 2
    
    x0 = _hex_initial_guess()
    
    bounds = [(0, 1), (0, h)] * N
    
    # 1. Global Search with Differential Evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-7,
            polish=False,
            init='sobol',
            popsize=20
        )
        best_points = result.x.reshape(-1, 2)
    except Exception:
        best_points = x0.reshape(-1, 2)
    
    # 2. Local Refinement with Powell
    if _is_valid(best_points):
        res = minimize(
            _objective,
            best_points.flatten(),
            method='Powell',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if _is_valid(res.x.reshape(-1, 2)):
            best_points = res.x.reshape(-1, 2)
            
    # 3. Shake and Refine
    rng = np.random.RandomState(42)
    current_best = best_points.copy()
    current_val = -_objective(best_points.flatten())
    
    for _ in range(10):
        # Jiggle
        noise = rng.uniform(-0.02, 0.02, size=current_best.shape)
        perturbed = current_best + noise
        
        # Ensure validity by reverting invalid points to original
        for i in range(len(perturbed)):
            if not _inside_triangle(perturbed[i]):
                perturbed[i] = current_best[i]
                
        res = minimize(
            _objective,
            perturbed.flatten(),
            method='Powell',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if _is_valid(res.x.reshape(-1, 2)):
            new_points = res.x.reshape(-1, 2)
            new_val = -_objective(new_points.flatten())
            if new_val > current_val:
                current_best = new_points
                current_val = new_val
                
    return current_best
# EVOLVE-BLOCK-END
