import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)

def get_triangle_indices(n):
    """Get all unique triplets (i, j, k) with i < j < k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices)

TRI_INDICES = get_triangle_indices(N)
N_TRIS = len(TRI_INDICES)  # 165

def compute_oriented_areas(points):
    """Compute oriented (signed) areas for all triangles.
    Returns array of shape (N_TRIS,) where area = 0.5 * cross product.
    """
    i_idx = TRI_INDICES[:, 0]
    j_idx = TRI_INDICES[:, 1]
    k_idx = TRI_INDICES[:, 2]
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    # Oriented area = 0.5 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    areas = 0.5 * (
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty_constraint(points):
    """Penalty for points outside the triangle"""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def nqp_objective(flat_params):
    """
    Objective for NQP formulation.
    flat_params: [alpha, x1, y1, x2, y2, ..., x11, y11]
    We maximize alpha subject to s_ijk * area_ijk >= alpha for all triangles.
    Using penalty method: minimize -alpha + penalty_for_violations + constraint_penalty
    """
    alpha = flat_params[0]
    points = flat_params[1:].reshape(N, 2)
    
    # Compute oriented areas
    areas = compute_oriented_areas(points)
    
    # For the NQP, we need to determine signs. 
    # In the penalty approach, we treat this as:
    # maximize alpha such that min_ijk |area_ijk| >= alpha
    # This is equivalent to: maximize min(areas) where areas are absolute values
    # But we handle it via the sign variables implicitly.
    
    # The effective constraint is: for each triangle, |area| >= alpha
    # Violation = max(0, alpha - |area|)
    abs_areas = np.abs(areas)
    violations = np.maximum(0, alpha - abs_areas)
    violation_penalty = np.sum(violations**2)
    
    # Constraint penalty for points outside triangle
    cpen = penalty_constraint(points)
    
    # Objective: maximize alpha, penalize violations and constraint violations
    # We want to push alpha up while keeping violations small
    obj = -alpha + 100.0 * violation_penalty + 1000.0 * cpen
    
    return obj

def nqp_gradient(flat_params):
    """
    Compute gradient of the NQP objective.
    """
    alpha = flat_params[0]
    points = flat_params[1:].reshape(N, 2)
    
    i_idx = TRI_INDICES[:, 0]
    j_idx = TRI_INDICES[:, 1]
    k_idx = TRI_INDICES[:, 2]
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    areas = 0.5 * (
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    abs_areas = np.abs(areas)
    
    # Gradient w.r.t. alpha
    grad_alpha = -1.0 + 200.0 * np.sum(np.maximum(0, alpha - abs_areas))
    
    # Gradient w.r.t. points
    grad_points = np.zeros((N, 2))
    
    # For each triangle, compute gradient of violation penalty
    # violation = max(0, alpha - |area|)
    # d(violation)/d(area) = -sign(area) * max(0, alpha - |area|) / |area| if |area| > 0
    # Actually: d(max(0, alpha - |a|))/da = -sign(a) * max(0, alpha - |a|)
    
    active = alpha > abs_areas
    if np.any(active):
        # For active violations, compute gradient
        # d|area|/d(points) for each triangle
        # area = 0.5 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
        # d(area)/dx1 = 0.5*(y2-y3), d(area)/dy1 = 0.5*(x3-x2)
        # d(area)/dx2 = 0.5*(y3-y1), d(area)/dy2 = 0.5*(x1-x3)
        # d(area)/dx3 = 0.5*(y1-y2), d(area)/dy3 = 0.5*(x2-x1)
        
        for t in range(N_TRIS):
            if not active[t]:
                continue
            
            i, j, k = i_idx[t], j_idx[t], k_idx[t]
            violation = alpha - abs_areas[t]
            
            # Gradient of area w.r.t. each point
            darea_dx1 = 0.5 * (p2[t, 1] - p3[t, 1])
            darea_dy1 = 0.5 * (p3[t, 0] - p2[t, 0])
            darea_dx2 = 0.5 * (p3[t, 1] - p1[t, 1])
            darea_dy2 = 0.5 * (p1[t, 0] - p3[t, 0])
            darea_dx3 = 0.5 * (p1[t, 1] - p2[t, 1])
            darea_dy3 = 0.5 * (p2[t, 0] - p1[t, 0])
            
            # d|area|/d(area) = sign(area)
            sign_a = np.sign(areas[t])
            if sign_a == 0:
                sign_a = 1.0
            
            # d(violation)/d(area) = -sign_a * violation (when violation > 0)
            # d(violation)/d(point) = d(violation)/d(area) * d(area)/d(point)
            
            factor = -sign_a * violation * 200.0  # 200 from 2*100
            
            grad_points[i, 0] += factor * darea_dx1
            grad_points[i, 1] += factor * darea_dy1
            grad_points[j, 0] += factor * darea_dx2
            grad_points[j, 1] += factor * darea_dy2
            grad_points[k, 0] += factor * darea_dx3
            grad_points[k, 1] += factor * darea_dy3
    
    # Gradient of constraint penalty
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    # d(pen)/dx, d(pen)/dy for each constraint
    # pen = sum(max(0, -d)^2)
    # d(max(0, -d)^2)/d(d) = -2*max(0, -d)
    
    for c in range(3):
        if c == 0:
            dc_dx = np.zeros(N)
            dc_dy = np.ones(N)
        elif c == 1:
            dc_dx = SQRT3 * np.ones(N)
            dc_dy = -np.ones(N)
        else:
            dc_dx = -SQRT3 * np.ones(N)
            dc_dy = -np.ones(N)
        
        d = [d1, d2, d3][c]
        viol = np.maximum(0, -d)
        if np.any(viol > 0):
            grad_points[:, 0] += 2000.0 * (-2 * viol) * dc_dx
            grad_points[:, 1] += 2000.0 * (-2 * viol) * dc_dy
    
    grad = np.zeros_like(flat_params)
    grad[0] = grad_alpha
    grad[1:] = grad_points.flatten()
    
    return grad

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    def evaluate(points):
        """Evaluate the score (minimum triangle area) for given points"""
        if np.any(points[:, 0] < -0.1) or np.any(points[:, 0] > 1.1) or \
           np.any(points[:, 1] < -0.1) or np.any(points[:, 1] > SQRT3/2 + 0.1):
            return -np.inf
        areas = compute_oriented_areas(points)
        min_area = np.min(np.abs(areas))
        cpen = penalty_constraint(points)
        if cpen > 1e-4:
            return min_area - 1000 * cpen
        return min_area
    
    def custom_gradient_descent(init_points, lr=0.01, max_iter=5000, alpha_init=0.001):
        """Custom gradient descent for the NQP formulation"""
        flat = np.zeros(1 + 2*N)
        flat[0] = alpha_init
        flat[1:] = init_points.flatten()
        
        best_flat = flat.copy()
        best_obj = nqp_objective(flat)
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            obj = nqp_objective(flat)
            if obj < best_obj:
                best_obj = obj
                best_flat = flat.copy()
            
            grad = nqp_gradient(flat)
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            
            # Normalize gradient for stable updates
            lr_eff = lr / (1 + 0.1 * grad_norm)
            flat = flat - lr_eff * grad
            
            # Project alpha to be non-negative
            flat[0] = max(0, flat[0])
            
            # Project points to stay in reasonable bounds
            points = flat[1:].reshape(N, 2)
            points[:, 0] = np.clip(points[:, 0], -0.5, 1.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, SQRT3/2 + 0.5)
            flat[1:] = points.flatten()
        
        # Return the best found
        best_points_out = best_flat[1:].reshape(N, 2)
        return best_points_out
    
    # Try multiple random restarts with custom gradient descent
    for seed in range(50):
        if time.time() - start_time > time_limit - 15:
            break
        
        rng = np.random.RandomState(seed)
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        
        # Run custom gradient descent
        result_points = custom_gradient_descent(init_points, lr=0.05, max_iter=3000, alpha_init=0.005)
        
        score = evaluate(result_points)
        if score > best_score:
            best_score = score
            best_points = result_points.copy()
    
    # Also try with scipy's L-BFGS-B as a refinement
    if best_points is not None and time.time() - start_time < time_limit - 20:
        def scipy_obj(flat_p):
            points = flat_p.reshape(N, 2)
            areas = compute_oriented_areas(points)
            min_area = np.min(np.abs(areas))
            cpen = penalty_constraint(points)
            return -min_area + 1000.0 * cpen
        
        def scipy_grad(flat_p):
            points = flat_p.reshape(N, 2)
            i_idx = TRI_INDICES[:, 0]
            j_idx = TRI_INDICES[:, 1]
            k_idx = TRI_INDICES[:, 2]
            p1 = points[i_idx]
            p2 = points[j_idx]
            p3 = points[k_idx]
            areas = 0.5 * (
                p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                p3[:, 0] * (p1[:, 1] - p2[:, 1])
            )
            abs_areas = np.abs(areas)
            
            # Find the minimum area triangle
            min_idx = np.argmin(abs_areas)
            i, j, k = i_idx[min_idx], j_idx[min_idx], k_idx[min_idx]
            
            # Gradient of -min_area
            # Only the minimum triangle contributes to the gradient
            sign_a = np.sign(areas[min_idx])
            if sign_a == 0:
                sign_a = 1.0
            
            darea_dx1 = 0.5 * (p2[min_idx, 1] - p3[min_idx, 1])
            darea_dy1 = 0.5 * (p3[min_idx, 0] - p2[min_idx, 0])
            darea_dx2 = 0.5 * (p3[min_idx, 1] - p1[min_idx, 1])
            darea_dy2 = 0.5 * (p1[min_idx, 0] - p3[min_idx, 0])
            darea_dx3 = 0.5 * (p1[min_idx, 1] - p2[min_idx, 1])
            darea_dy3 = 0.5 * (p2[min_idx, 0] - p1[min_idx, 0])
            
            grad_points = np.zeros((N, 2))
            grad_points[i, 0] = -sign_a * darea_dx1
            grad_points[i, 1] = -sign_a * darea_dy1
            grad_points[j, 0] = -sign_a * darea_dx2
            grad_points[j, 1] = -sign_a * darea_dy2
            grad_points[k, 0] = -sign_a * darea_dx3
            grad_points[k, 1] = -sign_a * darea_dy3
            
            # Add constraint penalty gradient
            x = points[:, 0]
            y = points[:, 1]
            d1 = y
            d2 = SQRT3 * x - y
            d3 = SQRT3 - SQRT3 * x - y
            
            for c in range(3):
                if c == 0:
                    dc_dx = np.zeros(N)
                    dc_dy = np.ones(N)
                elif c == 1:
                    dc_dx = SQRT3 * np.ones(N)
                    dc_dy = -np.ones(N)
                else:
                    dc_dx = -SQRT3 * np.ones(N)
                    dc_dy = -np.ones(N)
                
                d = [d1, d2, d3][c]
                viol = np.maximum(0, -d)
                if np.any(viol > 0):
                    grad_points[:, 0] += 2000.0 * (-2 * viol) * dc_dx
                    grad_points[:, 1] += 2000.0 * (-2 * viol) * dc_dy
            
            return grad_points.flatten()
        
        try:
            res = minimize(
                scipy_obj,
                best_points.flatten(),
                jac=scipy_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfev': 50000}
            )
            refined_points = res.x.reshape(N, 2)
            refined_score = evaluate(refined_points)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: symmetric configuration
        h = SQRT3 / 2
        best_points = np.array([
            [0.5, h/3],       # centroid
            [0.25, h/6], [0.75, h/6],
            [0.25, h/2], [0.75, h/2],
            [0.5, 2*h/3],
            [0.1, h/12], [0.9, h/12],
            [0.1, 5*h/12], [0.9, 5*h/12],
            [0.5, h/12]
        ])
    
    return best_points
