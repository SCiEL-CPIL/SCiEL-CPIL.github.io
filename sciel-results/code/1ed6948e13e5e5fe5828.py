"""Rewritten program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hull_area(p):
    """Return area of the convex hull of point set p."""
    try:
        hull = ConvexHull(p)
        return hull.volume
    except Exception:
        return 1e-10

def _all_triangle_areas(p):
    """Return array of 2*area for all triples."""
    n = len(p)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = abs(np.cross(p[j] - p[i], p[k] - p[i]))
                areas.append(a)
    return np.array(areas)

def _min_triangle_area(p):
    """Return minimum triangle area (2*area) over all triples."""
    return np.min(_all_triangle_areas(p))

def _score(p):
    """Return min_triangle_area / convex_hull_area."""
    hull_area = _hull_area(p)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(p)
    return min_area / hull_area

def _soft_obj(x, T=0.005):
    """
    Smooth lower bound on the minimum triangle area using log-sum-exp.
    We want to maximize the min area, so we minimize -logsumexp(-areas/T).
    
    Let A be the vector of all triangle areas (2*area).
    min(A) ≈ -T * log(sum(exp(-A/T)))
    So we want to maximize this, i.e., minimize -min(A) ≈ T * log(sum(exp(-A/T))).
    """
    p = x.reshape(-1, 2)
    areas = _all_triangle_areas(p)
    areas = np.maximum(areas, 1e-10)
    # Use log-sum-exp for numerical stability
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - max_a) / T))) + (-max_a / T)
    return T * log_sum

def _soft_obj_grad(x, T=0.005):
    """
    Gradient of the soft objective with respect to x.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    areas = _all_triangle_areas(p)
    areas = np.maximum(areas, 1e-10)
    
    # Compute weights: w_i = exp(-A_i / T) / sum(exp(-A_j / T))
    # For numerical stability
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / T)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Gradient of T * log(sum(exp(-A_i/T))) with respect to p
    # d/dp [T * log(sum(exp(-A_i/T)))] = T * (1/sum_exp) * sum(d/dp exp(-A_i/T))
    # d/dp exp(-A_i/T) = exp(-A_i/T) * (-1/T) * dA_i/dp
    # So gradient = T * (1/sum_exp) * sum(exp(-A_i/T) * (-1/T) * dA_i/dp)
    #             = -(1/sum_exp) * sum(exp(-A_i/T) * dA_i/dp)
    #             = -sum(weights_i * dA_i/dp)
    
    grad = np.zeros_like(x)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                w = weights[idx]
                # dA/dp for triangle (i, j, k)
                # A = |cross(p[j]-p[i], p[k]-p[i])|
                # For simplicity, assume the sign is positive (points in convex position)
                # dA/dp[i] = -cross(p[j]-p[i], p[k]-p[i])_perp ... 
                # Actually, let's use the formula:
                # A = 0.5 * |det([p_j - p_i, p_k - p_i])|
                # Let v1 = p[j] - p[i], v2 = p[k] - p[i]
                # A = 0.5 * |v1_x * v2_y - v1_y * v2_x|
                # dA/dp[i] = -0.5 * (v2_y, -v2_x) ... wait
                # Let's use the cross product derivative.
                # cross(u, v) = u_x v_y - u_y v_x
                # d/dp[i] cross(p[j]-p[i], p[k]-p[i]) 
                # = d/dp[i] ((p[j]-p[i])_x (p[k]-p[i])_y - (p[j]-p[i])_y (p[k]-p[i])_x)
                # Let u = p[j]-p[i], v = p[k]-p[i]
                # du/dp[i] = -I, dv/dp[i] = -I
                # d cross(u,v)/dp[i] = (du/dp[i])_x v_y + u_x (dv/dp[i])_y - (du/dp[i])_y v_x - u_y (dv/dp[i])_x
                # = (-1)v_y + u_x(-1) - (-1)v_x - u_y(-1)
                # = -v_y - u_x + v_x + u_y
                # This is a scalar. But we need the gradient vector.
                
                # Let's use a more direct approach for the gradient of the area.
                # The area of triangle with vertices a, b, c is 0.5 * |det([b-a, c-a])|
                # Assuming convex position, we can drop the absolute value and the 0.5 factor
                # (since we're just looking at the relative gradient).
                # A = det([b-a, c-a]) = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
                # dA/da_x = -(c_y - a_y) - (b_y - a_y) = -c_y + a_y - b_y + a_y = 2a_y - b_y - c_y
                # dA/da_y = -(b_x - a_x) + (c_x - a_x) = -b_x + a_x + c_x - a_x = c_x - b_x
                # dA/db_x = (c_y - a_y)
                # dA/db_y = -(c_x - a_x)
                # dA/dc_x = -(b_y - a_y)
                # dA/dc_y = (b_x - a_x)
                
                a = p[i]
                b = p[j]
                c = p[k]
                
                da_dx = 2*a[0] - b[0] - c[0]
                da_dy = c[1] - b[1]
                db_dx = c[1] - a[1]
                db_dy = -(c[0] - a[0])
                dc_dx = -(b[1] - a[1])
                dc_dy = b[0] - a[0]
                
                grad[2*i] += -w * da_dx
                grad[2*i+1] += -w * da_dy
                grad[2*j] += -w * db_dx
                grad[2*j+1] += -w * db_dy
                grad[2*k] += -w * dc_dx
                grad[2*k+1] += -w * dc_dy
                
                idx += 1
    
    return grad

def _two_ring_params_to_pts(params):
    """
    Convert two-ring parameters to points.
    params: [r1, theta1_0, ..., theta1_{K1-1}, r2, theta2_0, ..., theta2_{K2-1}]
    K1 + K2 = N
    """
    K1 = 7
    K2 = 7
    r1 = params[0]
    thetas1 = params[1:1+K1]
    r2 = params[1+K1]
    thetas2 = params[2+K1:2+K1+K2]
    
    pts1 = np.column_stack([r1 * np.cos(thetas1), r1 * np.sin(thetas1)])
    pts2 = np.column_stack([r2 * np.cos(thetas2), r2 * np.sin(thetas2)])
    
    pts = np.vstack([pts1, pts2])
    return pts

def _optimize_two_ring(x0, deadline):
    """Optimize the two-ring parameters."""
    def obj(params):
        if time.time() > deadline:
            return 1e10
        pts = _two_ring_params_to_pts(params)
        return _soft_obj(pts.flatten(), T=0.005)
    
    def grad(params):
        if time.time() > deadline:
            return np.zeros_like(params)
        pts = _two_ring_params_to_pts(params)
        g = _soft_obj_grad(pts.flatten(), T=0.005)
        # Chain rule: g_params = J^T g_pts
        # This is complex, so we'll use numerical gradient or just use L-BFGS-B without gradient
        # For simplicity, let's not provide gradient here and let L-BFGS-B estimate it
        return None
    
    result = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    if result.success:
        pts = _two_ring_params_to_pts(result.x)
        return pts
    return None

def _optimize_full_points(x0, deadline):
    """Optimize the full point coordinates."""
    def obj(x):
        if time.time() > deadline:
            return 1e10
        return _soft_obj(x, T=0.005)
    
    def grad(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        return _soft_obj_grad(x, T=0.005)
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(
        obj,
        x0,
        jac=grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
    )
    
    if result.success:
        return result.x.reshape(-1, 2)
    return None

def _initial_regular_polygon():
    """Return a regular 14-gon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for the 150s limit
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Two-ring optimization
    # Initial two-ring: 7 points on outer ring, 7 on inner ring
    K1 = 7
    K2 = 7
    r1_init = 1.0
    r2_init = 0.5
    thetas1_init = np.linspace(0, 2 * np.pi, K1, endpoint=False)
    thetas2_init = np.linspace(0, 2 * np.pi, K2, endpoint=False) + np.pi / K2  # Offset
    
    x0_two_ring = np.concatenate([[r1_init], thetas1_init, [r2_init], thetas2_init])
    
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        
        # Add some noise to the initial parameters
        noise = rng.normal(0, 0.1, size=x0_two_ring.shape)
        x0_noisy = x0_two_ring + noise
        x0_noisy[0] = np.clip(x0_noisy[0], 0.5, 1.5)  # r1
        x0_noisy[1+K1] = np.clip(x0_noisy[1+K1], 0.1, 1.0)  # r2
        
        pts = _optimize_two_ring(x0_noisy, deadline)
        if pts is not None:
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # Phase 2: Full-point optimization from the best two-ring result
    if best_points is not None:
        x0_full = best_points.flatten()
        for restart in range(2):
            if time.time() > deadline - 5.0:
                break
            
            noise = rng.normal(0, 0.05, size=x0_full.shape)
            x0_noisy = x0_full + noise
            
            pts = _optimize_full_points(x0_noisy, deadline)
            if pts is not None:
                score = _score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
    
    # Phase 3: Full-point optimization from regular polygon
    x0_reg = _initial_regular_polygon().flatten()
    for restart in range(2):
        if time.time() > deadline - 2.0:
            break
        
        noise = rng.normal(0, 0.1, size=x0_reg.shape)
        x0_noisy = x0_reg + noise
        
        pts = _optimize_full_points(x0_noisy, deadline)
        if pts is not None:
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # If no valid points found, return a default configuration
    if best_points is None:
        best_points = _initial_regular_polygon()
    
    return best_points
# EVOLVE-BLOCK-END
