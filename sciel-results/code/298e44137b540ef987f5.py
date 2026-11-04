"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2

def _min_triangle_area_vec(points):
    """Vectorized calculation of the minimum triangle area."""
    n = len(points)
    min_area = float('inf')
    # We can speed this up by precomputing differences, but for N=11, 
    # the triple loop is 165 iterations, which is very fast.
    # Vectorization over all triples:
    # Indices i < j < k
    idx_i, idx_j, idx_k = np.indices((n, n, n))
    # Filter i < j < k
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i_idx = idx_i[mask]
    j_idx = idx_j[mask]
    k_idx = idx_k[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    x_diff_ji = p_j[:, 0] - p_i[:, 0]
    y_diff_ji = p_j[:, 1] - p_i[:, 1]
    x_diff_ki = p_k[:, 0] - p_i[:, 0]
    y_diff_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = x_diff_ji * y_diff_ki - y_diff_ji * x_diff_ki
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _in_triangle_vec(points, eps=1e-9):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # y >= 0
    # x >= 0, x <= 1
    # y <= H * x (for x < 0.5) -> y - H*x <= 0
    # y <= H * (1-x) (for x > 0.5) -> y - H + H*x <= 0
    
    # Actually, the lines are:
    # Left edge: y = H * x  => y - H*x <= 0
    # Right edge: y = H * (1-x) => y + H*x - H <= 0
    # Bottom: y >= 0
    
    c1 = y - H * x
    c2 = y + H * x - H
    c3 = -y
    
    # All must be <= eps
    return np.all(c1 <= eps) and np.all(c2 <= eps) and np.all(c3 <= eps)

def _penalty(params):
    """Penalty for points outside the triangle."""
    points = params.reshape(N, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance outside
    # Bottom
    p_bottom = np.maximum(0, -y)
    # Left: y > H*x => y - H*x > 0
    p_left = np.maximum(0, y - H * x)
    # Right: y > H*(1-x) => y + H*x - H > 0
    p_right = np.maximum(0, y + H * x - H)
    
    # Sum of squares of violations
    penalty = np.sum(p_bottom**2) + np.sum(p_left**2) + np.sum(p_right**2)
    return penalty

def _objective_de(params):
    """Objective for Differential Evolution: minimize -min_area + penalty."""
    points = params.reshape(N, 2)
    pen = _penalty(params)
    if pen > 1e-6:
        # If significantly outside, return large value
        # But we want to guide it back. 
        # A large constant penalty helps DE reject bad points.
        return 100.0 + pen
    min_area = _min_triangle_area_vec(points)
    return -min_area + pen

def _objective_local(params):
    """Objective for local refinement: minimize -min_area + large penalty."""
    points = params.reshape(N, 2)
    pen = _penalty(params)
    # Use a very high penalty multiplier to ensure feasibility
    return -min_triangle_area_vec(points) + 1000.0 * pen

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    np.random.seed(42)
    
    # Bounds for DE: slightly larger than the triangle to allow exploration
    # Triangle x in [0, 1], y in [0, H]
    bounds = [(-0.1, 1.1), (-0.1, H + 0.1)] * N
    
    # Run Differential Evolution
    # maxiter controls generations, popsize controls population size
    # For 22 dimensions, popsize=20 is ~440 points per gen.
    # 100 gens = 44000 evaluations. Each eval is ~165 area calcs.
    # This should be fast enough.
    try:
        result_de = differential_evolution(
            _objective_de, bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-8,
            polish=False, # We will polish manually
            init='latinhypercube'
        )
        best_params = result_de.x
    except Exception:
        # Fallback to random if DE fails
        best_params = np.random.uniform(0, 1, size=(N, 2))
        # Project to triangle roughly
        for i in range(N):
            p = best_params[i]
            if p[1] > H * min(p[0], 1-p[0]):
                p[1] = H * min(p[0], 1-p[0]) * 0.9
            if p[1] < 0: p[1] = 0.01
            if p[0] < 0: p[0] = 0.01
            if p[0] > 1: p[0] = 0.99

    # Local refinement using L-BFGS-B
    # We use a penalty method because L-BFGS-B doesn't support non-linear inequality constraints easily
    # (it supports bounds, but the triangle is not a box).
    # We can use SLSQP for local refinement as it handles constraints better.
    
    def obj_slsqp(params):
        points = params.reshape(N, 2)
        min_area = _min_triangle_area_vec(points)
        return -min_area

    constraints = []
    bounds = []
    for i in range(N):
        # x_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i]})
        bounds.append((0, 1))
        # y_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1]})
        bounds.append((0, H))
        # y_i <= H * x_i
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: H * p[2*i] - p[2*i+1]})
        # y_i <= H * (1 - x_i)
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: H * (1 - p[2*i]) - p[2*i+1]})

    # Run SLSQP on the best result from DE
    try:
        result_local = minimize(
            obj_slsqp, best_params, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if result_local.success or result_local.fun < -_min_triangle_area_vec(best_params.reshape(N,2)):
             best_params = result_local.x
    except Exception:
        pass

    # Final check and projection if slightly out due to numerical errors
    points = best_params.reshape(N, 2)
    # Project any points that are slightly outside
    for i in range(N):
        x, y = points[i]
        if y < 0: y = 0
        if x < 0: x = 0
        if x > 1: x = 1
        max_y = H * min(x, 1-x)
        if y > max_y: y = max_y
        points[i] = [x, y]
        
    return points
