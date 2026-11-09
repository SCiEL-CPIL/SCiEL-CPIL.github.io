"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay


def _area_triangle(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Minimum area of any triangle formed by three of the given points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area_triangle(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _project_to_triangle(x, y):
    """Project point (x,y) onto the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    h = np.sqrt(3) / 2
    # Barycentric coordinates for triangle with vertices A=(0,0), B=(1,0), C=(0.5, h)
    # Using the standard barycentric coordinate formula
    v0 = np.array([1.0, 0.0]) - np.array([x, y])
    v1 = np.array([1.0, 0.0]) - np.array([0.5, h])
    v2 = np.array([0.5, h]) - np.array([0.0, 0.0])
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d02 = np.dot(v0, v2)
    d12 = np.dot(v1, v2)
    d22 = np.dot(v2, v2)
    
    denom = d00 * d22 - d02 * d02
    if denom == 0:
        return np.array([0.0, 0.0])
    
    a = (d11 * d02 - d01 * d12) / denom
    b = (d00 * d12 - d01 * d02) / denom
    
    if a < 0:
        a = 0
    if b < 0:
        b = 0
    if a + b > 1:
        s = a + b
        a /= s
        b /= s
    
    # Reconstruct point
    px = a * 0.0 + b * 1.0 + (1 - a - b) * 0.5
    py = a * 0.0 + b * 0.0 + (1 - a - b) * h
    return np.array([px, py])


def _inside_triangle(x, y, eps=1e-9):
    """Check if point (x,y) is inside or on the boundary of the equilateral triangle."""
    h = np.sqrt(3) / 2
    # The triangle has vertices (0,0), (1,0), (0.5, h)
    # Edge from (0,0) to (1,0): y >= -eps
    # Edge from (0,0) to (0.5, h): y <= sqrt(3)*x + eps
    # Edge from (1,0) to (0.5, h): y <= -sqrt(3)*(x-1) + eps = sqrt(3)*(1-x) + eps
    if y < -eps:
        return False
    if y > np.sqrt(3) * x + eps:
        return False
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True


def _objective(params):
    """Objective function to minimize: negative of the minimum triangle area."""
    h = np.sqrt(3) / 2
    points = []
    for i in range(N):
        x = params[2 * i]
        y = params[2 * i + 1]
        # Project to triangle if outside
        if not _inside_triangle(x, y, eps=-1e-12):
            pt = _project_to_triangle(x, y)
            x, y = pt[0], pt[1]
        points.append([x, y])
    points = np.array(points)
    
    min_area = _min_triangle_area(points)
    return -min_area


def _local_polish(points, max_iter=100, tol=1e-10):
    """Local optimization using coordinate descent / Nelder-Mead style refinement."""
    points = np.array(points, dtype=float)
    h = np.sqrt(3) / 2
    
    def objective_local(params):
        pts = []
        for i in range(N):
            x = params[2 * i]
            y = params[2 * i + 1]
            if not _inside_triangle(x, y, eps=-1e-12):
                pt = _project_to_triangle(x, y)
                x, y = pt[0], pt[1]
            pts.append([x, y])
        pts = np.array(pts)
        return -_min_triangle_area(pts)
    
    best_params = points.flatten()
    best_val = objective_local(best_params)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            for dim in range(2):
                # Try small perturbations
                for step in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
                    for sign in [1, -1]:
                        trial = best_params.copy()
                        trial[2 * i + dim] += sign * step
                        val = objective_local(trial)
                        if val < best_val - tol:
                            best_params = trial
                            best_val = val
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_params.reshape(N, 2), -best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s safety margin
    
    h = np.sqrt(3) / 2
    
    # Define bounds for each point
    bounds = [(-0.1, 1.1), (-0.1, h + 0.1)] * N
    
    best_points = None
    best_score = 0.0
    
    # Multiple restarts with differential evolution
    n_restarts = 8
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42 + restart,
                init='sobol',
                polish=True,
                workers=1
            )
            
            if time.time() > deadline - 15:
                break
            
            params = result.x
            points = []
            for i in range(N):
                x = params[2 * i]
                y = params[2 * i + 1]
                if not _inside_triangle(x, y, eps=-1e-12):
                    pt = _project_to_triangle(x, y)
                    x, y = pt[0], pt[1]
                points.append([x, y])
            points = np.array(points)
            
            score = _min_triangle_area(points)
            
            # Local polish
            if time.time() < deadline - 10:
                polished_params, polished_score = _local_polish(points, max_iter=50, tol=1e-11)
                polished_points = []
                for i in range(N):
                    x = polished_params[2 * i]
                    y = polished_params[2 * i + 1]
                    if not _inside_triangle(x, y, eps=-1e-12):
                        pt = _project_to_triangle(x, y)
                        x, y = pt[0], pt[1]
                    polished_points.append([x, y])
                polished_points = np.array(polished_points)
                
                if polished_score > score:
                    points = polished_points
                    score = polished_score
            
            if score > best_score:
                best_score = score
                best_points = points
                
        except Exception:
            continue
    
    # If no valid result from DE, use a simple grid-based initialization
    if best_points is None:
        # Use a symmetric configuration as fallback
        # Place points in a hexagonal-like pattern
        best_points = np.array([
            [0.5, h * 0.5],  # center
            [0.15, h * 0.2],
            [0.85, h * 0.2],
            [0.5, h * 0.85],
            [0.3, h * 0.5],
            [0.7, h * 0.5],
            [0.1, h * 0.6],
            [0.9, h * 0.6],
            [0.35, h * 0.15],
            [0.65, h * 0.15],
            [0.5, h * 0.3]
        ])
        # Project all to triangle
        for i in range(N):
            if not _inside_triangle(best_points[i, 0], best_points[i, 1], eps=-1e-12):
                pt = _project_to_triangle(best_points[i, 0], best_points[i, 1])
                best_points[i] = pt
    
    return best_points
# EVOLVE-BLOCK-END
