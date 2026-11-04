"""Heilbronn triangles: maximize smallest triangle area among 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
_DEADLINE = 150.0  # seconds; evaluator kills at 150s, keep margin
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START) - 2.0  # 2s safety margin


def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


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


def _inside_triangle(pt, eps=1e-9):
    """Check if point is inside or on boundary of the unit equilateral triangle."""
    x, y = pt
    if y < -eps:
        return False
    if y > np.sqrt(3) * x + eps:
        return False
    if y > -np.sqrt(3) * (x - 1) + eps:
        return False
    return True


def _project_to_triangle(pt):
    """Project point onto the unit equilateral triangle."""
    x, y = pt
    # Clip to triangle using barycentric-like clipping
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # Constraints: y >= 0, y <= sqrt(3)*x, y <= -sqrt(3)*(x-1)
    if y < 0:
        y = 0
    if y > np.sqrt(3) * x:
        # Project onto line y = sqrt(3)*x
        # The line from A to C: direction (0.5, sqrt(3)/2)
        # Simplest: reduce x to make y = sqrt(3)*x, or reduce y
        x = y / np.sqrt(3)
    if y > -np.sqrt(3) * (x - 1):
        # Project onto line from B to C: y = -sqrt(3)*(x-1)
        # Solve for x given y: x = 1 - y/sqrt(3)
        x = 1 - y / np.sqrt(3)
    # Final clamp
    x = max(0.0, min(1.0, x))
    y = max(0.0, y)
    if y > np.sqrt(3) * x:
        y = np.sqrt(3) * x
    if y > -np.sqrt(3) * (x - 1):
        y = -np.sqrt(3) * (x - 1)
    return np.array([x, y])


def _objective_and_constraints(x_flat, t):
    """
    Objective: minimize -t (to maximize t)
    Constraints: for each triple (i,j,k), area^2 >= t^2
    i.e., area^2 - t^2 >= 0
    
    x_flat: 2*N array of point coordinates
    t: scalar
    """
    points = x_flat.reshape(N, 2)
    n = N
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas[i, j, k] = a
                areas[j, i, k] = a
                areas[i, k, j] = a
                areas[k, j, i] = a
                areas[j, k, i] = a
                areas[k, i, j] = a
    
    return points, areas


def _run_qp_optimization(x0_2d, t0, max_iter=200):
    """
    Run trust-constr optimization to maximize t subject to area^2 >= t^2 for all triples.
    x0_2d: (N, 2) initial points
    t0: initial t value
    """
    x0 = np.concatenate([x0_2d.flatten(), [t0]])
    
    def objective(x):
        points = x[:2*N].reshape(N, 2)
        t = x[2*N]
        # We want to maximize t, so minimize -t
        return -t
    
    def objective_grad(x):
        g = np.zeros_like(x)
        g[2*N] = -1.0
        return g
    
    # Build constraint functions
    triples = list(itertools.combinations(range(N), 3))
    
    def make_constraint(i, j, k):
        def constraint(x):
            points = x[:2*N].reshape(N, 2)
            t = x[2*N]
            a = _triangle_area(points[i], points[j], points[k])
            return a * a - t * t
        return constraint
    
    def make_constraint_grad(i, j, k):
        def constraint_grad(x):
            points = x[:2*N].reshape(N, 2)
            t = x[2*N]
            p1, p2, p3 = points[i], points[j], points[k]
            # area = 0.5 * |cross|, but for gradient we use signed area
            # Let's use the unsigned area squared
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            a = 0.5 * abs(cross)
            
            if a < 1e-12:
                return np.zeros_like(x)
            
            # d(a^2)/dx = 2*a * da/dx
            # da/dx where a = 0.5*|cross|
            # For simplicity, use subgradient: sign(cross) * 0.5 * dcross/dx
            sgn = 1.0 if cross >= 0 else -1.0
            da_dx = 0.5 * sgn * np.zeros_like(x)
            
            # dcross/dx1: cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            # d/dx1 = -(y3-y1) + (y2-y1)*(1) ... let me be careful
            # cross = x2*y3 - x2*y1 - x1*y3 + x1*y1 - y2*x3 + y2*x1 + y1*x3 - y1*x1
            # = x2*y3 - x2*y1 - x1*y3 + x1*y1 - y2*x3 + y2*x1 + y1*x3 - y1*x1
            # Simplify: x1*y1 - y1*x1 = 0
            # = x2*y3 - x2*y1 - x1*y3 - y2*x3 + y2*x1 + y1*x3
            # d/dx1 = -y3 + y2
            # d/dy1 = -x2 + x3
            # d/dx2 = y3 - y1
            # d/dy2 = -x3 + x1
            # d/dx3 = -y2 + y1
            # d/dy3 = x2 - x1
            
            idx1, idx2, idx3 = 2*i, 2*j, 2*k
            da_dx[idx1] += 0.5 * sgn * (-p3[1] + p2[1])
            da_dx[idx1+1] += 0.5 * sgn * (-p2[0] + p3[0])
            da_dx[idx2] += 0.5 * sgn * (p3[1] - p1[1])
            da_dx[idx2+1] += 0.5 * sgn * (-p3[0] + p1[0])
            da_dx[idx3] += 0.5 * sgn * (-p2[1] + p1[1])
            da_dx[idx3+1] += 0.5 * sgn * (p2[0] - p1[0])
            
            dconstraint_dx = 2 * a * da_dx
            dconstraint_dx[2*N] = -2 * t
            
            return dconstraint_dx
        return constraint_grad
    
    constraints = []
    for (i, j, k) in triples:
        constraints.append({
            'type': 'ineq',
            'fun': make_constraint(i, j, k),
            'jac': make_constraint_grad(i, j, k)
        })
    
    # Bounds: points inside triangle, t >= 0
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, np.sqrt(3)/2))  # y
    bounds.append((0.0, 0.5))  # t upper bound (area of whole triangle is sqrt(3)/4 ≈ 0.433)
    
    try:
        result = minimize(
            objective,
            x0,
            method='trust-constr',
            jac=objective_grad,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': max_iter, 'gtol': 1e-6, 'xtol': 1e-8}
        )
        if result.success or result.fun < 0:
            x_opt = result.x
            # Project points to triangle
            points_opt = x_opt[:2*N].reshape(N, 2)
            for i in range(N):
                points_opt[i] = _project_to_triangle(points_opt[i])
            t_opt = max(0.0, x_opt[2*N])
            return points_opt, t_opt
    except Exception:
        pass
    
    return x0_2d, t0


def _local_refinement(points):
    """Local refinement: for each point, try small perturbations to improve min area."""
    points = points.copy()
    improved = True
    while improved:
        improved = False
        current_min = _min_triangle_area(points)
        for i in range(N):
            best_pt = points[i].copy()
            best_min = current_min
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    new_pt = points[i] + np.array([dx, dy])
                    if _inside_triangle(new_pt):
                        test_pts = points.copy()
                        test_pts[i] = new_pt
                        test_min = _min_triangle_area(test_pts)
                        if test_min > best_min:
                            best_min = test_min
                            best_pt = new_pt
            if not np.allclose(best_pt, points[i]):
                points[i] = best_pt
                current_min = best_min
                improved = True
    return points, _min_triangle_area(points)


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Multi-start: try several initial configurations
    rng = np.random.RandomState(42)
    
    # Start 1: Random points
    for trial in range(5):
        if _time_left() < 10:
            break
        
        # Generate random points inside triangle
        x0 = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    x0[i] = [u + 0.5 * v, (np.sqrt(3) / 2) * v]
                    break
        
        # Initial t: use current min area as lower bound
        t0 = _min_triangle_area(x0) * 0.9
        
        # Run QP optimization
        points_opt, t_opt = _run_qp_optimization(x0, t0, max_iter=100)
        
        # Local refinement
        points_refined, score = _local_refinement(points_opt)
        
        if score > best_score:
            best_score = score
            best_points = points_refined.copy()
        
        print(f"Trial {trial}: score={score:.6f}, t_opt={t_opt:.6f}", flush=True)
    
    # Final local refinement on best
    if best_points is not None:
        best_points, best_score = _local_refinement(best_points)
    
    if best_points is None:
        # Fallback: simple random
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    points[i] = [u + 0.5 * v, (np.sqrt(3) / 2) * v]
                    break
        return points
    
    return best_points
