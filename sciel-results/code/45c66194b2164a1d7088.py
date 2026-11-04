import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _triangle_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns a 1D array of areas.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 0.0

def _soft_min(values, tau):
    """
    Compute the soft minimum of a set of values.
    -1/tau * log(sum(exp(-tau * v)))
    """
    # Shift for numerical stability
    min_val = np.min(values)
    shifted = values - min_val
    # tau * shifted is positive, so -tau * shifted is negative
    # We want sum(exp(-tau * v)) = exp(-tau * min_val) * sum(exp(-tau * (v - min_val)))
    # log(sum(exp(-tau * v))) = -tau * min_val + log(sum(exp(-tau * (v - min_val))))
    # soft_min = min_val - 1/tau * log(sum(exp(-tau * (v - min_val))))
    
    # To avoid overflow in exp, we use the fact that shifted >= 0.
    # exp(-tau * shifted) <= 1. So sum is <= len(values). No overflow.
    # However, for large tau, exp(-tau * shifted) underflows to 0 for shifted > 0.
    # This is fine, it effectively picks the minimum.
    
    exp_terms = np.exp(-tau * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return min_val
    
    log_sum = np.log(sum_exp)
    return min_val - (1.0 / tau) * log_sum

def _param_to_points(params):
    """
    Convert parameters to points.
    Params: [r_0, ..., r_12, delta_0, ..., delta_11]
    r_i > 0
    delta_i > 0, sum(delta) = 2*pi
    """
    r = params[:N]
    # Ensure r is positive. We optimize log(r) or use bounds.
    # Let's use bounds in minimize, so r is already positive.
    # But to be safe, clip.
    r = np.maximum(r, 1e-6)
    
    deltas = params[N:N + N - 1]
    # Ensure deltas are positive
    deltas = np.maximum(deltas, 1e-6)
    
    # Normalize deltas to sum to 2*pi
    sum_deltas = np.sum(deltas)
    if sum_deltas < 1e-10:
        deltas = np.ones(N - 1) * (2 * np.pi / (N - 1))
    else:
        deltas = deltas * (2 * np.pi / sum_deltas)
    
    # Compute angles
    # theta_0 = 0
    # theta_i = sum(delta_0 ... delta_{i-1})
    thetas = np.zeros(N)
    thetas[1:] = np.cumsum(deltas)
    # thetas[N-1] is sum of first N-1 deltas. The last delta is implicitly 2*pi - thetas[N-1].
    # Wait, we have N points. We need N angles.
    # The gaps are between p0-p1, p1-p2, ..., p11-p12, p12-p0.
    # We optimized N-1 gaps. The last gap is determined.
    # Let's construct angles:
    # theta_0 = 0
    # theta_1 = delta_0
    # theta_2 = delta_0 + delta_1
    # ...
    # theta_{N-1} = sum(delta_0 ... delta_{N-2})
    # The last gap is 2*pi - theta_{N-1}.
    
    # This parametrization covers all convex polygons with p0 at angle 0.
    # Due to rotational symmetry, fixing theta_0=0 is fine.
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    
    return np.column_stack((x, y))

def _objective(params, tau):
    """
    Objective function to minimize.
    Returns - (soft_min_triangle_area / hull_area)
    """
    points = _param_to_points(params)
    
    # Check for degeneracy
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    areas = _triangle_areas(points)
    
    # If any area is 0, soft min will be 0, objective 0. We want to avoid this.
    # But soft min handles it.
    
    hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 1e10
        
    min_area_soft = _soft_min(areas, tau)
    
    ratio = min_area_soft / hull_area
    
    return -ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Regular 13-gon
    # r_i = 1
    # delta_i = 2*pi/13
    r_init = np.ones(N)
    delta_init = np.full(N - 1, 2 * np.pi / N)
    params_init = np.concatenate((r_init, delta_init))
    
    best_params = params_init.copy()
    best_score = 0.0
    
    # Evaluate initial
    # For exact score, we use tau -> infinity equivalent (exact min)
    points_init = _param_to_points(params_init)
    areas_init = _triangle_areas(points_init)
    hull_init = _convex_hull_area(points_init)
    if hull_init > 1e-10:
        best_score = np.min(areas_init) / hull_init
    
    # Optimization schedule
    # We use increasing tau to sharpen the minimum
    taus = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    # Bounds for params
    # r_i in [0.1, 10.0]
    # delta_i in [0.1, 6.0] (approx pi/2 is max reasonable gap for convexity without being weird, but 2pi is limit)
    # Actually, delta_i can be up to 2*pi - (N-1)*min_delta.
    # Let's use wide bounds.
    bounds = []
    for _ in range(N):
        bounds.append((0.1, 10.0))
    for _ in range(N - 1):
        bounds.append((0.1, 6.0))
    
    current_params = params_init.copy()
    
    for tau in taus:
        if time.time() > deadline:
            break
            
        try:
            # L-BFGS-B is good for bounded problems
            res = minimize(
                _objective, 
                current_params, 
                args=(tau,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < _objective(current_params, tau):
                current_params = res.x
            
            # Evaluate exact score for this candidate
            points_curr = _param_to_points(current_params)
            areas_curr = _triangle_areas(points_curr)
            hull_curr = _convex_hull_area(points_curr)
            
            if hull_curr > 1e-10:
                score_curr = np.min(areas_curr) / hull_curr
                if score_curr > best_score:
                    best_score = score_curr
                    best_params = current_params.copy()
                    
        except Exception as e:
            # If optimization fails, keep previous
            pass
            
        # Time check
        if time.time() > deadline:
            break
            
    # Final evaluation of best_params
    best_points = _param_to_points(best_params)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print("Constructed points:", pts.shape)
