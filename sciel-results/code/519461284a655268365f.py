"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle with area 1/2 (side length 1)
# The problem says "unit-area triangle" but the interface specifies vertices (0,0), (1,0), (1/2, sqrt(3)/2)
# This triangle has area sqrt(3)/4 ≈ 0.433. The evaluator divides by its area.
# So we work in this coordinate system and the objective is to maximize min triangle area / (sqrt(3)/4)

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric-like check:
    # y >= 0
    # y <= sqrt(3)/2 * x  (for x <= 0.5)
    # y <= sqrt(3)/2 * (1 - x)  (for x >= 0.5)
    if y < -eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    max_y = (3 ** 0.5) / 2 * min(x, 1 - x)
    if y > max_y + eps:
        return False
    return True

def _project_to_triangle(p):
    """Project a point onto the equilateral triangle."""
    x, y = p
    # Simple clipping approach
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    max_y = (3 ** 0.5) / 2 * min(x, 1 - x)
    if y > max_y:
        y = max_y
    return np.array([x, y])

def _objective_and_constraints(params):
    """
    params: 2N array of point coordinates.
    Returns: negative of min triangle area (to minimize), and constraint violations.
    """
    points = params.reshape(N, 2)
    
    # Check if all points are in the triangle
    for i in range(N):
        if not _in_triangle(points[i], eps=1e-7):
            # Penalty for being outside
            penalty = 0
            x, y = points[i]
            if y < 0:
                penalty += abs(y)
            if x < 0:
                penalty += abs(x)
            if x > 1:
                penalty += abs(x - 1)
            max_y = (3 ** 0.5) / 2 * min(max(x, 0), 1 - max(x, 0))
            if y > max_y:
                penalty += abs(y - max_y)
            return penalty * 100 + 100, points
    
    min_area = _min_triangle_area(points)
    return -min_area, points

def _run_optimization(initial_points, max_iter=5000, tol=1e-10):
    """Run optimization to maximize min triangle area."""
    params0 = initial_points.flatten()
    
    # Use SLSQP with numerical gradients
    # We minimize -min_area, subject to points being in the triangle
    
    best_params = params0.copy()
    best_val = _min_triangle_area(initial_points)
    
    # Multi-start approach
    for restart in range(3):
        if restart > 0:
            # Perturb the current best
            params0 = best_params + np.random.normal(0, 0.05, size=params0.shape)
        
        # Define the objective: minimize -min_area
        def obj(params):
            val, _ = _objective_and_constraints(params)
            return val
        
        # Constraints: each point must be in the triangle
        # We'll handle this with a penalty in the objective for simplicity,
        # but let's use inequality constraints for the triangle boundaries
        
        def constraint_y_nonneg(i):
            return lambda p: p[2*i+1]
        
        def constraint_x_nonneg(i):
            return lambda p: p[2*i]
        
        def constraint_x_leq_1(i):
            return lambda p: 1 - p[2*i]
        
        def constraint_slope_left(i):
            # y <= sqrt(3)/2 * x  for the left edge
            return lambda p: (3 ** 0.5) / 2 * p[2*i] - p[2*i+1]
        
        def constraint_slope_right(i):
            # y <= sqrt(3)/2 * (1-x) for the right edge
            return lambda p: (3 ** 0.5) / 2 * (1 - p[2*i]) - p[2*i+1]
        
        constraints = []
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': constraint_y_nonneg(i)})
            constraints.append({'type': 'ineq', 'fun': constraint_x_nonneg(i)})
            constraints.append({'type': 'ineq', 'fun': constraint_x_leq_1(i)})
            constraints.append({'type': 'ineq', 'fun': constraint_slope_left(i)})
            constraints.append({'type': 'ineq', 'fun': constraint_slope_right(i)})
        
        bounds = [(0, 1)] * (2 * N)
        
        try:
            result = minimize(
                obj, params0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
            )
            if result.success or result.fun < obj(params0):
                points_res = result.x.reshape(N, 2)
                # Verify all points are in triangle
                valid = True
                for i in range(N):
                    if not _in_triangle(points_res[i], eps=1e-6):
                        valid = False
                        break
                if valid:
                    val = _min_triangle_area(points_res)
                    if val > best_val:
                        best_val = val
                        best_params = result.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(N, 2), best_val

def _initial_points_grid():
    """Generate initial points using a structured grid-like approach."""
    # Start with a roughly uniform distribution
    points = []
    # Place points at various positions
    # Use a simple heuristic: divide the triangle into smaller regions
    # For 11 points, we can use a combination of vertices, edge midpoints, and interior points
    
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, (3 ** 0.5) / 2])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, (3 ** 0.5) / 4])
    points.append([0.75, (3 ** 0.5) / 4])
    
    # Interior points (roughly uniform)
    # Use barycentric coordinates for uniform sampling
    for _ in range(5):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        x = u * 0.5 + v * 0.5  # This isn't quite right for barycentric
        # Proper barycentric: p = b1*v1 + b2*v2 + b3*v3, b1+b2+b3=1
        b1 = 1 - u - v
        b2 = u
        b3 = v
        x = b1 * 0 + b2 * 1 + b3 * 0.5
        y = b1 * 0 + b2 * 0 + b3 * (3 ** 0.5) / 2
        points.append([x, y])
    
    return np.array(points)

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Config 1: Grid-like
    initial_configs.append(_initial_points_grid())
    
    # Config 2: Random
    for _ in range(2):
        pts = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            b1 = 1 - u - v
            b2 = u
            b3 = v
            x = b1 * 0 + b2 * 1 + b3 * 0.5
            y = b1 * 0 + b2 * 0 + b3 * (3 ** 0.5) / 2
            pts.append([x, y])
        initial_configs.append(np.array(pts))
    
    # Config 3: Symmetric pattern
    # Place points in a hexagonal-like pattern
    pts = []
    # Center
    pts.append([0.5, (3 ** 0.5) / 6])
    # Ring 1
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        r = 0.15
        x = 0.5 + r * np.cos(angle)
        y = (3 ** 0.5) / 6 + r * np.sin(angle)
        if _in_triangle([x, y]):
            pts.append([x, y])
    # Add more points if needed
    while len(pts) < N:
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        b1 = 1 - u - v
        b2 = u
        b3 = v
        x = b1 * 0 + b2 * 1 + b3 * 0.5
        y = b1 * 0 + b2 * 0 + b3 * (3 ** 0.5) / 2
        pts.append([x, y])
    initial_configs.append(np.array(pts[:N]))
    
    # Optimize each configuration and keep the best
    best_points = None
    best_val = -float('inf')
    
    for config in initial_configs:
        points, val = _run_optimization(config, max_iter=3000)
        if val > best_val:
            best_val = val
            best_points = points
    
    # Final verification
    if best_points is None:
        # Fallback
        best_points = initial_configs[0]
    
    # Ensure all points are strictly inside (with small margin)
    for i in range(N):
        best_points[i] = _project_to_triangle(best_points[i])
    
    return best_points
