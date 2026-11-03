"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0  # Area of the unit-side equilateral triangle


def _area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _all_areas(points):
    """Compute all C(n,3) triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_area(points[i], points[j], points[k]))
    return np.array(areas)


def _min_area(points):
    """Minimum triangle area among all triplets."""
    return np.min(_all_areas(points))


def _project_to_triangle(p):
    """Project a point into the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x, y = p
    # Check against three edges
    # Edge 1: y >= 0
    # Edge 2: y <= sqrt(3)*x (left edge from (0,0) to (0.5, sqrt(3)/2))
    # Edge 3: y <= sqrt(3)*(1-x) (right edge from (1,0) to (0.5, sqrt(3)/2))
    
    # For the equilateral triangle with vertices (0,0), (1,0), (0.5, h) where h=sqrt(3)/2:
    # The constraints are:
    # y >= 0
    # y <= sqrt(3) * x  (for x <= 0.5)
    # y <= sqrt(3) * (1 - x)  (for x >= 0.5)
    
    # More precisely, the three edges are:
    # Bottom: from (0,0) to (1,0): y >= 0
    # Left: from (0,0) to (0.5, sqrt(3)/2): y <= sqrt(3)*x
    # Right: from (1,0) to (0.5, sqrt(3)/2): y <= sqrt(3)*(1-x)
    
    h = np.sqrt(3) / 2.0
    
    # Project using barycentric-like clipping
    # The triangle can be described as: y >= 0, y <= sqrt(3)*min(x, 1-x)
    
    if y < 0:
        y = 0.0
    max_y = np.sqrt(3) * min(x, 1.0 - x)
    if y > max_y:
        y = max_y
    if x < 0:
        x = 0.0
    if x > 1:
        x = 1.0
        y = 0.0
    return np.array([x, y])


def _objective_flat(flat_points):
    """Objective: negative minimum area (to be minimized)."""
    pts = flat_points.reshape(-1, 2)
    # Project each point into the triangle
    pts = np.array([_project_to_triangle(p) for p in pts])
    return -_min_area(pts), pts


def _gradient_descent_step(points, learning_rate=0.01):
    """One step of subgradient descent on the minimum area."""
    n = len(points)
    areas = _all_areas(points)
    min_idx = np.argmin(areas)
    
    # Find which triplet gives the minimum
    # Reconstruct indices
    idx = 0
    min_triplet = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if idx == min_idx:
                    min_triplet = (i, j, k)
                    break
                idx += 1
            if min_triplet:
                break
        if min_triplet:
            break
    
    if min_triplet is None:
        return points
    
    i, j, k = min_triplet
    p1, p2, p3 = points[i], points[j], points[k]
    
    # Subgradient of area w.r.t. each point
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    # dA/dp1 = -0.5 * cross(p2-p1, p3-p1) direction... 
    # Actually, let's use the formula:
    # A = 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
    # The sign matters. Let's compute the signed area and use its gradient.
    
    # Signed area: S = 0.5 * ((x2-x1)(y3-y1) - (x3-x1)(y2-y1))
    # dS/dx1 = 0.5 * (-(y3-y1) + (y2-y1)) = 0.5*(y2-y3)
    # dS/dy1 = 0.5 * ((x3-x1) - (x2-x1)) = 0.5*(x3-x2)
    # dS/dx2 = 0.5 * (y3-y1)
    # dS/dy2 = 0.5 * (-(x3-x1))
    # dS/dx3 = 0.5 * (-(y2-y1))
    # dS/dy3 = 0.5 * (x2-x1)
    
    sign = 1.0 if areas[min_idx] >= 0 else -1.0
    # We want to maximize the area, so we move in the direction that increases |S|
    # If S > 0, increase S; if S < 0, decrease S (i.e., increase |S|)
    
    grad = np.zeros_like(points)
    s = 0.5 * ((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    if s >= 0:
        # Increase S
        grad[i, 0] += 0.5 * (p2[1] - p3[1])
        grad[i, 1] += 0.5 * (p3[0] - p2[0])
        grad[j, 0] += 0.5 * (p3[1] - p1[1])
        grad[j, 1] += 0.5 * (-(p3[0] - p1[0]))
        grad[k, 0] += 0.5 * (-(p2[1] - p1[1]))
        grad[k, 1] += 0.5 * (p2[0] - p1[0])
    else:
        # Decrease S (increase |S|)
        grad[i, 0] -= 0.5 * (p2[1] - p3[1])
        grad[i, 1] -= 0.5 * (p3[0] - p2[0])
        grad[j, 0] -= 0.5 * (p3[1] - p1[1])
        grad[j, 1] -= 0.5 * (-(p3[0] - p1[0]))
        grad[k, 0] -= 0.5 * (-(p2[1] - p1[1]))
        grad[k, 1] -= 0.5 * (p2[0] - p1[0])
    
    new_points = points + learning_rate * grad
    # Project back
    new_points = np.array([_project_to_triangle(p) for p in new_points])
    return new_points


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial best: use a reasonable starting configuration
    # Start with points spread out
    rng = np.random.RandomState(42)
    
    # Try multiple random restarts
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points
        if restart == 0:
            # Symmetric initial placement
            points = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2],
                [0.25, np.sqrt(3)/4],
                [0.75, np.sqrt(3)/4],
                [0.5, np.sqrt(3)/2 * 0.5],
                [0.125, np.sqrt(3)/8],
                [0.375, np.sqrt(3)/8],
                [0.625, np.sqrt(3)/8],
                [0.875, np.sqrt(3)/8],
                [0.5, np.sqrt(3)/2 * 0.75],
            ])
        else:
            # Random initial points inside triangle
            points = []
            for _ in range(N):
                u, v = rng.random(), rng.random()
                if u + v > 1.0:
                    u, v = 1.0 - u, 1.0 - v
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2.0) * v
                points.append([x, y])
            points = np.array(points)
        
        current_score = _min_area(points) / TRI_AREA
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Use scipy minimize with multiple methods
        flat = points.flatten()
        
        def obj(x):
            pts = x.reshape(-1, 2)
            pts = np.array([_project_to_triangle(p) for p in pts])
            return -_min_area(pts)
        
        try:
            result = minimize(
                obj, flat, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() > deadline:
                break
            
            opt_points = np.array([_project_to_triangle(p) for p in result.x.reshape(-1, 2)])
            opt_score = _min_area(opt_points) / TRI_AREA
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
        
        # Also try L-BFGS-B with numerical gradient
        if time.time() > deadline:
            break
        try:
            result2 = minimize(
                obj, flat, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if time.time() > deadline:
                break
            
            opt_points2 = np.array([_project_to_triangle(p) for p in result2.x.reshape(-1, 2)])
            opt_score2 = _min_area(opt_points2) / TRI_AREA
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2.copy()
        except Exception:
            pass
        
        # Local refinement with subgradient steps
        if time.time() > deadline:
            break
        if best_points is not None:
            local_pts = best_points.copy()
            lr = 0.005
            for step in range(500):
                if time.time() > deadline:
                    break
                local_pts = _gradient_descent_step(local_pts, learning_rate=lr)
                local_score = _min_area(local_pts) / TRI_AREA
                if local_score > best_score:
                    best_score = local_score
                    best_points = local_pts.copy()
                # Decay learning rate
                lr *= 0.99
    
    if best_points is None:
        # Fallback
        best_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [0.25, np.sqrt(3)/4],
            [0.75, np.sqrt(3)/4],
            [0.5, np.sqrt(3)/4],
            [0.125, np.sqrt(3)/8],
            [0.375, np.sqrt(3)/8],
            [0.625, np.sqrt(3)/8],
            [0.875, np.sqrt(3)/8],
            [0.5, np.sqrt(3)/2 * 0.75],
        ])
    
    return best_points
