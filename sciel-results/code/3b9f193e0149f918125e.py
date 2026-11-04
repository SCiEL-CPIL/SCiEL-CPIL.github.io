"""Heilbronn triangles: maximize the smallest triangle area among 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 15.0  # seconds to reserve

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0  # height


def _barycentric_coords(pt):
    """Return barycentric coordinates (a, b, c) with a+b+c=1 for point pt in triangle ABC."""
    x, y = pt
    # Using the formula: for triangle with vertices A, B, C
    # a = ((C.y - A.y)*(x - A.x) + (A.x - C.x)*(y - A.y)) / (C.y - A.y)*(B.x - A.x) - (B.y - A.y)*(C.x - A.x)
    # Simpler: use area ratios
    # a is weight of A, b of B, c of C
    # a = 1 - b - c
    # b = (C.y*(x-A.x) - C.x*(y-A.y) + A.y*(C.x-B.x) - A.x*(C.y-B.y)) / ... 
    # Let's use the standard formula
    # For triangle A=(0,0), B=(1,0), C=(0.5, H):
    # c = y / H
    # b = x - 0.5 * y / H  ... wait, let me derive properly
    # Point = a*A + b*B + c*C, a+b+c=1
    # x = b*1 + c*0.5
    # y = c*H
    # So c = y/H
    # b = x - 0.5*c = x - 0.5*y/H
    # a = 1 - b - c
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.array([a, b, c])


def _inside_triangle(pt, tol=1e-9):
    """Check if point is inside or on the boundary of the equilateral triangle."""
    bc = _barycentric_coords(pt)
    return np.all(bc >= -tol) and np.all(bc <= 1.0 + tol)


def _project_to_triangle(pt):
    """Project point to the triangle if outside, using barycentric clamping."""
    bc = _barycentric_coords(pt)
    bc_clamped = np.clip(bc, 0.0, 1.0)
    # Normalize to sum to 1
    s = np.sum(bc_clamped)
    if s < 1e-12:
        return A.copy()
    bc_clamped /= s
    return bc_clamped[0] * A + bc_clamped[1] * B + bc_clamped[2] * C


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)


def _min_triangle_area(points):
    """Minimum triangle area among all triples."""
    return np.min(_all_triangle_areas(points))


def _soft_min(areas, beta):
    """Soft minimum using log-sum-exp for stability."""
    # soft_min = -log(sum(exp(-beta * a_i))) / beta
    # For numerical stability, use the max trick
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum_exp / beta


def _soft_min_grad(areas, beta):
    """Gradient of soft min with respect to areas."""
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_vals)
    # d(soft_min)/d(a_i) = -exp(-beta*a_i) / sum(exp(-beta*a_j)) * ... 
    # soft_min = -1/beta * log(sum(exp(-beta*a)))
    # d(soft_min)/da_i = -1/beta * (1/sum(exp(-beta*a))) * (-beta * exp(-beta*a_i))
    # = exp(-beta*a_i) / sum(exp(-beta*a_j))
    probs = exp_vals / sum_exp
    return probs


def _objective_and_grad(points, beta, time_budget=None):
    """
    Objective: minimize the negative soft-min of triangle areas.
    This is equivalent to maximizing the soft-min.
    Returns (objective_value, gradient).
    """
    n = len(points)
    points = np.array(points)
    
    # Compute all triangle areas
    areas = []
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                areas.append(area)
                triples.append((i, j, k, cross))
    
    areas = np.array(areas)
    soft_min_val = _soft_min(areas, beta)
    objective = -soft_min_val  # We want to maximize soft_min, so minimize negative
    
    # Gradient computation
    grad = np.zeros_like(points)
    probs = _soft_min_grad(areas, beta)
    
    for idx, ((i, j, k, cross), prob) in enumerate(zip(triples, probs)):
        p1, p2, p3 = points[i], points[j], points[k]
        # d(area)/d(p1), d(area)/d(p2), d(area)/d(p3)
        # area = 0.5 * |cross|
        # If cross > 0: d(area)/dp = 0.5 * d(cross)/dp
        # If cross < 0: d(area)/dp = -0.5 * d(cross)/dp
        # d(cross)/dp1 = -(p2-p3) rotated... let's compute directly
        # cross = (p2x-p1x)(p3y-p1y) - (p2y-p1y)(p3x-p1x)
        # d(cross)/dp1x = -(p3y-p1y) + (p2y-p1y) = p2y - p3y
        # d(cross)/dp1y = (p3x-p1x) - (p2x-p1x) = p3x - p2x
        # d(cross)/dp2x = (p3y-p1y)
        # d(cross)/dp2y = -(p3x-p1x)
        # d(cross)/dp3x = -(p2y-p1y)
        # d(cross)/dp3y = (p2x-p1x)
        
        sign = np.sign(cross)
        if sign == 0:
            sign = 1.0
        
        factor = 0.5 * sign * prob / beta  # Wait, let me re-derive
        # d(soft_min)/d(a_idx) = prob
        # d(objective)/d(a_idx) = -prob
        # d(objective)/dp = -prob * d(area_idx)/dp
        # d(area_idx)/dp = 0.5 * sign(cross) * d(cross)/dp
        
        # So d(objective)/dp = -prob * 0.5 * sign * d(cross)/dp
        
        d_cross_dp1 = np.array([p2[1] - p3[1], p3[0] - p2[0]])
        d_cross_dp2 = np.array([p3[1] - p1[1], -(p3[0] - p1[0])])
        d_cross_dp3 = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        
        grad[i] += -prob * 0.5 * sign * d_cross_dp1
        grad[j] += -prob * 0.5 * sign * d_cross_dp2
        grad[k] += -prob * 0.5 * sign * d_cross_dp3
    
    return objective, grad


def _optimize_from_start(points, beta, max_iter=500, tol=1e-8):
    """Optimize from a starting point using L-BFGS-B with projection."""
    points = np.array(points, dtype=float)
    
    def obj_wrapper(x):
        pts = x.reshape(-1, 2)
        # Project each point to triangle
        pts = np.array([_project_to_triangle(p) for p in pts])
        val, _ = _objective_and_grad(pts, beta)
        return val
    
    def grad_wrapper(x):
        pts = x.reshape(-1, 2)
        pts = np.array([_project_to_triangle(p) for p in pts])
        _, g = _objective_and_grad(pts, beta)
        return g.flatten()
    
    # Bounds for each coordinate
    bounds = [(0.0, 1.0)] * (2 * N)
    
    result = minimize(
        obj_wrapper,
        points.flatten(),
        jac=grad_wrapper,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
    )
    
    pts = result.x.reshape(-1, 2)
    pts = np.array([_project_to_triangle(p) for p in pts])
    return pts, result.fun


def _random_start(rng):
    """Generate a random starting configuration."""
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            pt = [u + 0.5 * v, H * v]
            points.append(pt)
    return np.array(points)


def _grid_start():
    """Generate a grid-like starting configuration."""
    points = []
    # Try a few different patterns
    # Pattern 1: rows
    for i in range(N):
        # Distribute points in a triangular grid
        row = i // 3
        col = i % 3
        if row < 3:
            x = 0.1 + 0.3 * col + 0.05 * row
            y = 0.1 + 0.25 * row
        else:
            x = 0.2 + 0.1 * (i - 3)
            y = 0.5 + 0.05 * (i - 3)
        pt = [x, y]
        if _inside_triangle(pt):
            points.append(pt)
        else:
            points.append(_project_to_triangle(pt))
    # Fill remaining
    rng = np.random.RandomState(42)
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            pt = [u + 0.5 * v, H * v]
            points.append(pt)
    return np.array(points[:N])


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple beta values for soft-min
    beta_values = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    # Initial configurations
    rng = np.random.RandomState(7)
    configs = []
    
    # Random starts
    for _ in range(5):
        configs.append(_random_start(rng))
    
    # Grid start
    configs.append(_grid_start())
    
    # Symmetric start: use known good patterns
    # Try a regular hexagon + center + 4 more
    sym_points = []
    # Center
    sym_points.append([0.5, H / 3.0])
    # Hexagon vertices
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        r = 0.25
        x = 0.5 + r * np.cos(angle)
        y = H / 3.0 + r * np.sin(angle)
        if _inside_triangle([x, y]):
            sym_points.append([x, y])
        else:
            sym_points.append(_project_to_triangle([x, y]))
    # Add more points
    while len(sym_points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            pt = [u + 0.5 * v, H * v]
            sym_points.append(pt)
    configs.append(np.array(sym_points[:N]))
    
    for beta in beta_values:
        if time.time() > deadline:
            break
        
        for config in configs:
            if time.time() > deadline:
                break
            
            try:
                pts, _ = _optimize_from_start(config, beta, max_iter=200, tol=1e-7)
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Final polish with higher beta and more iterations
    if best_points is not None and time.time() < deadline:
        for beta in [500.0, 1000.0]:
            if time.time() > deadline:
                break
            try:
                pts, _ = _optimize_from_start(best_points, beta, max_iter=300, tol=1e-9)
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return a random configuration
        best_points = _random_start(np.random.RandomState(7))
    
    return best_points
