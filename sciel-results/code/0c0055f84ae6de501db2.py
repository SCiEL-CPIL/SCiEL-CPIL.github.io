import numpy as np
from scipy.optimize import minimize
import time

def _clamp_to_triangle(pts):
    """
    Clamp points to be inside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  => x >= y / sqrt(3)
    # 3. y <= sqrt(3) * (1 - x) => x <= 1 - y / sqrt(3)
    
    # We need to project onto the intersection of half-planes.
    # A simple iterative clamping might not be perfect for corners, but for optimization 
    # penalties, we can just calculate the distance to the boundary for the penalty term.
    # However, if we use penalty, we don't strictly need to clamp the input to the optimizer, 
    # just penalize violations.
    # But for the final result, we must return valid points.
    
    # Let's implement a proper projection for the final result.
    # For the optimizer, we will use a penalty.
    
    # Projection algorithm for triangle:
    # 1. If y < 0: y = 0.
    # 2. If x < y/sqrt(3): x = y/sqrt(3).
    # 3. If x > 1 - y/sqrt(3): x = 1 - y/sqrt(3).
    # 4. Re-check y constraints if x changed? 
    #    Actually, the region is convex. The projection is the closest point.
    #    Simple sequential clamping works for this specific triangle shape if done carefully.
    
    # Let's use barycentric coordinates for exact projection.
    # Vertices: A(0,0), B(1,0), C(0.5, h) where h = sqrt(3)/2.
    # Barycentric coords (u, v, w) for point (x,y):
    # w = y / h
    # u = (x - 0.5*w) / (1 - 0.5) ? No.
    # Let's use the standard formula:
    # w = y / h
    # u = (x - 0.5*y/h) / (1 - 0.5) is wrong.
    
    # Area method:
    # Area(ABC) = 0.5 * 1 * h = sqrt(3)/4.
    # w = Area(PAB) / Area(ABC) = (0.5 * 1 * y) / (sqrt(3)/4) = 2*y/sqrt(3) = y/h.
    # u = Area(PBC) / Area(ABC).
    # v = Area(PCA) / Area(ABC).
    # u+v+w = 1.
    
    # If u,v,w >= 0, point is inside.
    # If any < 0, clamp to 0 and renormalize? No, that's not Euclidean projection.
    # But for our purposes, if we use a penalty in the optimizer, the points will stay close to the boundary.
    # For the final output, we can just clamp coordinates sequentially which is sufficient for "on or inside".
    
    # Sequential clamp:
    y = np.maximum(y, 0.0)
    x = np.maximum(x, y / np.sqrt(3))
    x = np.minimum(x, 1.0 - y / np.sqrt(3))
    # Check if x clamp violated y?
    # If x was clamped to y/sqrt(3), then y <= sqrt(3)x holds.
    # If x was clamped to 1-y/sqrt(3), then y <= sqrt(3)(1-x) holds.
    # So y is still >= 0 and <= boundaries.
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _triangle_area(p1, p2, p3):
    """Calculate area of triangle formed by p1, p2, p3."""
    # 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])

def _soft_min(values, beta):
    """Stable soft-min."""
    # values: array of areas
    # To avoid overflow in exp, subtract min
    v_min = np.min(values)
    exp_vals = np.exp(-beta * (values - v_min))
    log_sum = np.log(np.sum(exp_vals))
    return v_min - np.log_sum / beta

def _objective_and_grad(params, beta, penalty_weight):
    """
    params: shape (n, 2)
    Returns objective value and gradient.
    """
    n = params.shape[0]
    pts = params
    
    # Calculate all triangle areas
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append(_triangle_area(pts[i], pts[j], pts[k]))
    areas = np.array(areas)
    
    # Soft min of areas
    sm = _soft_min(areas, beta)
    
    # Penalty for points outside triangle
    # Distance to triangle boundaries
    # 1. y < 0
    # 2. y > sqrt(3) x
    # 3. y > sqrt(3) (1-x)
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    pen = 0.0
    grad_pen = np.zeros_like(pts)
    
    # Boundary 1: y >= 0
    viol1 = np.where(y < 0, -y, 0.0)
    pen += np.sum(viol1**2)
    # grad wrt y
    grad_pen[:, 1] += 2 * viol1 * (-1) # d/dy (y^2) where y<0 is 2y, but viol is -y, so d/dy (-y)^2 = 2(-y)(-1) = 2y? 
    # Let's be precise. Penalty term is (max(0, -y))^2.
    # If y < 0, term is y^2. d/dy = 2y.
    # If y >= 0, term is 0. d/dy = 0.
    # So grad_y += 2 * y * (y < 0)
    
    # Boundary 2: y <= sqrt(3) x  =>  sqrt(3) x - y >= 0
    val2 = np.sqrt(3) * x - y
    viol2 = np.where(val2 < 0, -val2, 0.0) # violation amount
    pen += np.sum(viol2**2)
    # d/dx: if val2 < 0, d/dx (-(sqrt(3)x - y))^2 = 2(-val2)(-sqrt(3)) = 2 val2 sqrt(3)
    # d/dy: if val2 < 0, d/dy (-(sqrt(3)x - y))^2 = 2(-val2)(1) = -2 val2
    mask2 = val2 < 0
    grad_pen[:, 0] += 2 * val2 * mask2 * np.sqrt(3)
    grad_pen[:, 1] += -2 * val2 * mask2
    
    # Boundary 3: y <= sqrt(3) (1-x) => sqrt(3)(1-x) - y >= 0
    val3 = np.sqrt(3) * (1 - x) - y
    viol3 = np.where(val3 < 0, -val3, 0.0)
    pen += np.sum(viol3**2)
    # d/dx: d/dx (-(sqrt(3)(1-x)-y))^2 = 2(-val3)(sqrt(3)) = -2 val3 sqrt(3)
    # d/dy: d/dy (-(sqrt(3)(1-x)-y))^2 = 2(-val3)(1) = -2 val3
    mask3 = val3 < 0
    grad_pen[:, 0] += -2 * val3 * mask3 * np.sqrt(3)
    grad_pen[:, 1] += -2 * val3 * mask3
    
    # Update penalty gradient
    grad_pen *= penalty_weight
    pen *= penalty_weight
    
    # Total objective: -soft_min(areas) + penalty
    obj = -sm + pen
    
    # Gradient of soft min
    # sm = v_min - (1/beta) log(sum(exp(-beta(v_i - v_min))))
    # Let S = sum(exp(-beta(v_i - v_min)))
    # dsm/dv_k = ...
    # It's easier to compute gradient numerically or analytically.
    # Analytical gradient of soft min:
    # weights w_i = exp(-beta * (v_i - v_min)) / S
    # dsm/dv_k = -w_k * (1/beta) * beta ? No.
    # sm = -1/beta log(sum e^{-beta v_i}) (assuming v_min=0 for a moment, shift doesn't change grad)
    # dsm/dv_k = -1/beta * (1/S) * (-beta e^{-beta v_k}) = w_k
    # So d(-sm)/dv_k = -w_k
    # But we have the shift v_min. The shift makes it stable but the gradient form is the same.
    # w_i = exp(-beta(v_i - v_min)) / sum(exp(-beta(v_j - v_min)))
    
    v_min = np.min(areas)
    exp_vals = np.exp(-beta * (areas - v_min))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    # Gradient of objective wrt points
    grad_obj = np.zeros_like(pts)
    
    # We need d(areas)/d(pts)
    # This is tedious to do analytically for all 165 triangles.
    # Given the small size (11 points), numerical gradient is feasible and robust.
    # However, 22 variables * 2 (for central diff) * 165 triangles * eval cost...
    # Eval cost is ~165 ops. 22*2*165 = 7260 ops. Very fast.
    
    # Let's use numerical gradient for the area part to avoid coding errors.
    eps = 1e-6
    for i in range(n):
        for d in range(2):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i, d] += eps
            params_minus[i, d] -= eps
            
            # Recalculate soft min for plus
            areas_plus = []
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        areas_plus.append(_triangle_area(params_plus[a], params_plus[b], params_plus[c]))
            areas_plus = np.array(areas_plus)
            sm_plus = _soft_min(areas_plus, beta)
            
            areas_minus = []
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        areas_minus.append(_triangle_area(params_minus[a], params_minus[b], params_minus[c]))
            areas_minus = np.array(areas_minus)
            sm_minus = _soft_min(areas_minus, beta)
            
            # Objective is -sm + penalty. Penalty gradient is handled analytically above.
            # So we only add the gradient from the -sm part here.
            grad_obj[i, d] = (-sm_plus + sm_minus) / (2 * eps)
            
    total_grad = grad_obj + grad_pen
    return obj, total_grad

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    n_points = 11
    # Initialize points
    # Random points in triangle
    # Use rejection sampling or barycentric random
    # u, v ~ Uniform(0,1)
    # if u+v > 1, v = 1-u
    # Point = u*V1 + v*V2 + (1-u-v)*V0
    
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3)/2.0])
    
    pts = np.zeros((n_points, 2))
    for i in range(n_points):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            v = 1 - u
        pts[i] = u * V1 + v * V2 # V0 is 0
        
    # Add center point and corners? 
    # Better to start with a heuristic layout:
    # 1 center, 3 corners, 3 midpoints, 3 more?
    # Let's just use random init + optimization.
    
    # Optimization
    beta = 20.0
    penalty_weight = 100.0
    
    # Flatten for scipy
    x0 = pts.flatten()
    
    def fun(x):
        p = x.reshape(n_points, 2)
        val, _ = _objective_and_grad(p, beta, penalty_weight)
        return val
    
    def grad(x):
        p = x.reshape(n_points, 2)
        _, g = _objective_and_grad(p, beta, penalty_weight)
        return g.flatten()
        
    # Use L-BFGS-B
    # We don't have box constraints on x,y directly that map to the triangle, 
    # so we rely on the penalty.
    # However, L-BFGS-B might wander far away. 
    # Let's add loose bounds to keep points in [0, 1] x [0, 1]
    # The triangle is within this square.
    bounds = [(0, 1) for _ in range(2 * n_points)]
    
    best_val = np.inf
    best_x = x0
    
    # Run optimization
    try:
        res = minimize(fun, x0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'gtol': 1e-5, 'disp': False})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        # Try a few different betas if time permits
        for b in [50.0, 10.0]:
            if time.time() > deadline - 10:
                break
            beta = b
            res2 = minimize(fun, x0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'gtol': 1e-5, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception as e:
        pass
        
    # Final clamp
    final_pts = best_x.reshape(n_points, 2)
    final_pts = _clamp_to_triangle(final_pts)
    
    return final_pts
