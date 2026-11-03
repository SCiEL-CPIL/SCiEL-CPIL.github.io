"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
# Vertices of the equilateral triangle with side length 1
# Area = sqrt(3)/4
SQRT3 = 3 ** 0.5
TRI_AREA = SQRT3 / 4.0

# Vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def _all_triangles_areas(points):
    """Compute areas of all C(N,3) triangles formed by the points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area(points):
    """Minimum area among all triangles formed by the points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, margin=0.0):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -margin:
        return False
    if x < -margin:
        return False
    if x > 1.0 + margin:
        return False
    # Check below the top edge: y <= sqrt(3) * x for left half, y <= sqrt(3) * (1-x) for right half
    # Actually, the three edges are:
    # Bottom: y >= 0
    # Left: y <= sqrt(3) * x  (line from (0,0) to (0.5, sqrt(3)/2))
    # Right: y <= sqrt(3) * (1 - x)  (line from (1,0) to (0.5, sqrt(3)/2))
    if y > SQRT3 * x + margin:
        return False
    if y > SQRT3 * (1.0 - x) + margin:
        return False
    return True

def _project_to_triangle(p):
    """Project a point onto the equilateral triangle if it's outside."""
    x, y = p
    # If outside, we need to clip to the triangle
    # Simple approach: use barycentric coordinates
    # For triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates (a, b, c) where a+b+c=1
    # p = a*V0 + b*V1 + c*V2
    # x = b + c/2
    # y = c*sqrt(3)/2
    # So c = y / (sqrt(3)/2) = 2y/sqrt(3)
    # b = x - c/2 = x - y/sqrt(3)
    # a = 1 - b - c
    c = 2.0 * y / SQRT3
    b = x - c / 2.0
    a = 1.0 - b - c
    
    # Clip to valid barycentric coordinates
    a = max(0.0, a)
    b = max(0.0, b)
    c = max(0.0, c)
    s = a + b + c
    if s > 0:
        a /= s
        b /= s
        c /= s
    else:
        a, b, c = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    
    return a * V0 + b * V1 + c * V2

def _objective(params):
    """Objective function: negative of the minimum triangle area."""
    points = params.reshape(N, 2)
    # Project each point to the triangle
    points = np.array([_project_to_triangle(p) for p in points])
    min_area = _min_triangle_area(points)
    return -min_area

def _soft_min_objective(params, beta):
    """Soft-min objective for smoother optimization."""
    points = params.reshape(N, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    areas = _all_triangles_areas(points)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, shift by min
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    shifted = np.clip(shifted, -500, 500)
    log_sum = min_a + np.log(np.sum(np.exp(shifted))) / beta
    return -log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    def try_optimize(init_points, beta_schedule, max_iters=200):
        nonlocal best_points, best_score
        current_time = time.time()
        if current_time > deadline - 5.0:
            return
        
        points = np.array(init_points, dtype=float)
        # Ensure all points are inside the triangle
        points = np.array([_project_to_triangle(p) for p in points])
        
        # Check initial score
        init_score = _min_triangle_area(points)
        if init_score > best_score:
            best_score = init_score
            best_points = points.copy()
        
        # Try with soft-min annealing
        for beta in beta_schedule:
            current_time = time.time()
            if current_time > deadline - 5.0:
                break
            
            def obj(p):
                return _soft_min_objective(p, beta)
            
            try:
                res = minimize(obj, points.ravel(), method='L-BFGS-B', 
                              options={'maxiter': max_iters, 'ftol': 1e-12})
                if res.success or res.fun < 0:
                    opt_points = res.x.reshape(N, 2)
                    opt_points = np.array([_project_to_triangle(p) for p in opt_points])
                    score = _min_triangle_area(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
                    points = opt_points
            except Exception:
                pass
        
        # Final polish with hard min
        current_time = time.time()
        if current_time > deadline - 3.0:
            return
        
        def obj_hard(p):
            return _objective(p)
        
        try:
            res = minimize(obj_hard, points.ravel(), method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14})
            if res.fun < 0:
                opt_points = res.x.reshape(N, 2)
                opt_points = np.array([_project_to_triangle(p) for p in opt_points])
                score = _min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # Generate diverse initial configurations
    rng = np.random.RandomState(42)
    
    # 1. Random uniform in triangle
    for _ in range(3):
        u, v = rng.random(), rng.random()
        while u + v >= 1.0:
            u, v = rng.random(), rng.random()
        init = np.array([[u + 0.5 * v, SQRT3 / 2 * v] for _ in range(N)])
        # Add some random jitter
        init += rng.uniform(-0.05, 0.05, size=(N, 2))
        try_optimize(init, [20, 50, 100, 200], max_iters=150)
    
    # 2. Grid-like initial patterns
    # Triangular lattice points
    for scale in [0.8, 0.9, 1.0]:
        # Generate points on a triangular grid inside the triangle
        init = []
        for i in range(4):
            for j in range(4 - i):
                x = 0.1 + 0.25 * i * scale + 0.125 * j * scale
                y = 0.1 + 0.25 * SQRT3 / 2 * i * scale
                if _is_inside_triangle([x, y], 0.02):
                    init.append([x, y])
                if len(init) >= N:
                    break
            if len(init) >= N:
                break
        if len(init) < N:
            # Fill remaining with random
            while len(init) < N:
                u, v = rng.random(), rng.random()
                while u + v >= 1.0:
                    u, v = rng.random(), rng.random()
                init.append([u + 0.5 * v, SQRT3 / 2 * v])
        init = np.array(init[:N])
        try_optimize(init, [20, 50, 100, 200], max_iters=150)
    
    # 3. Symmetric patterns
    # Points along the median and symmetric pairs
    for base_y in [0.2, 0.3, 0.4]:
        init = []
        # Center point on median
        init.append([0.5, base_y])
        # Symmetric pairs
        for dy in [0.1, 0.2, 0.3]:
            y = base_y + dy
            if y < SQRT3 / 2 - 0.05:
                # Find x range at this y
                x_left = y / SQRT3
                x_right = 1.0 - y / SQRT3
                if x_right > x_left:
                    mid_x = (x_left + x_right) / 2
                    spread = (x_right - x_left) * 0.3
                    init.append([mid_x - spread, y])
                    init.append([mid_x + spread, y])
        # Add more points if needed
        while len(init) < N:
            u, v = rng.random(), rng.random()
            while u + v >= 1.0:
                u, v = rng.random(), rng.random()
            init.append([u + 0.5 * v, SQRT3 / 2 * v])
        init = np.array(init[:N])
        try_optimize(init, [20, 50, 100, 200], max_iters=150)
    
    # 4. Perturbations of best found so far
    current_time = time.time()
    if best_points is not None and current_time < deadline - 10.0:
        for _ in range(5):
            current_time = time.time()
            if current_time > deadline - 5.0:
                break
            perturbed = best_points + rng.uniform(-0.1, 0.1, size=(N, 2))
            try_optimize(perturbed, [50, 100, 200], max_iters=100)
    
    # Return the best result
    if best_points is None:
        # Fallback: random points
        points = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            while u + v >= 1.0:
                u, v = rng.random(), rng.random()
            points.append([u + 0.5 * v, SQRT3 / 2 * v])
        best_points = np.array(points)
    
    # Ensure all points are inside the triangle
    best_points = np.array([_project_to_triangle(p) for p in best_points])
    
    return best_points
