import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _cartesian_to_spherical(points):
    """Convert Nx3 Cartesian points to Nx2 Spherical (theta, phi)."""
    # points are normalized
    z = points[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return np.column_stack([theta, phi])

def _spherical_to_cartesian(params):
    """Convert Nx2 Spherical params to Nx3 Cartesian points."""
    theta = params[:, 0]
    phi = params[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _covering_radius(points, test_points):
    """
    Calculate the covering radius of 'points' given a set of 'test_points'.
    Returns the max angular distance from any test point to its nearest neighbor in 'points'.
    """
    # points: Nx3, test_points: Mx3
    # Compute dot products
    dots = test_points @ points.T
    # Clip for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distances
    dists = np.arccos(dots)
    # For each test point, find min distance to any config point
    min_dists = np.min(dists, axis=1)
    # Covering radius is the max of these min distances
    return np.max(min_dists)

def _smooth_objective(params, test_points, beta=10.0):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    params: 2N array of (theta, phi) pairs.
    """
    points = _spherical_to_cartesian(params.reshape(N, 2))
    
    # Compute distances from all test points to all config points
    dots = test_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # MxN
    
    # min over N for each test point
    min_dists = np.min(dists, axis=1) # M
    
    # Log-sum-exp approximation of max
    # max(x) ~= 1/beta * log(sum(exp(beta*x)))
    # To avoid overflow, subtract max
    max_d = np.max(min_dists)
    if max_d == 0:
        return 0.0
    exp_dists = np.exp(beta * (min_dists - max_d))
    log_sum = np.log(np.sum(exp_dists))
    return (max_d + log_sum / beta)

def _objective_and_gradient(params, test_points, beta=10.0):
    """
    Compute objective and gradient.
    """
    points = _spherical_to_cartesian(params.reshape(N, 2))
    
    # Distances
    dots = test_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # MxN
    
    # Find nearest neighbor index for each test point
    min_indices = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(test_points)), min_indices]
    
    # Objective
    max_d = np.max(min_dists)
    if max_d == 0:
        return 0.0, np.zeros_like(params)
    
    weights = np.exp(beta * (min_dists - max_d))
    sum_weights = np.sum(weights)
    obj = max_d + np.log(sum_weights) / beta
    
    # Gradient calculation is complex. 
    # d/dp_i arccos(y . p_i) = -y / sqrt(1 - (y.p_i)^2)
    # But we only need gradient for the nearest neighbor of each test point.
    # And the log-sum-exp adds a weighting.
    
    # Let's use finite difference or a simpler gradient-free method if gradient is too hard.
    # Given time constraints, L-BFGS-B without gradient (using finite diff) might be too slow for 100 variables.
    # Let's try Powell (derivative-free, more robust to non-smoothness if we use the raw max).
    # Or Nelder-Mead.
    
    # Actually, let's just return the objective and let scipy use finite differences if we don't provide grad.
    # But for 100 variables, finite diff is expensive.
    # Let's implement the gradient.
    
    # grad_obj = sum_{y} w_y * d/dp (min_dist(y))
    # w_y = exp(beta*(min_dist(y) - max_d)) / sum_weights
    # d/dp_i min_dist(y) is non-zero only if p_i is the nearest neighbor of y.
    # Let I_y be the index of the nearest neighbor for test point y.
    # d/dp_{I_y} arccos(y . p_{I_y}) = -y / sqrt(1 - (y . p_{I_y})^2)
    
    grad_cart = np.zeros_like(points) # Nx3
    
    # Accumulate gradients
    for idx, (y, p_idx, w) in enumerate(zip(test_points, min_indices, weights)):
        p = points[p_idx]
        dot = np.dot(y, p)
        # Gradient of arccos(dot) w.r.t p is -y / sqrt(1-dot^2)
        denom = np.sqrt(np.clip(1.0 - dot*dot, 1e-12, None))
        grad_p = -y / denom
        grad_cart[p_idx] += w * grad_p
    
    # Now we need to transform gradient from Cartesian to Spherical coordinates.
    # This is the tricky part.
    # x = sin(t)cos(p), y = sin(t)sin(p), z = cos(t)
    # dx/dt = cos(t)cos(p), dx/dp = -sin(t)sin(p)
    # dy/dt = cos(t)sin(p), dy/dp = sin(t)cos(p)
    # dz/dt = -sin(t),      dz/dp = 0
    
    # grad_t = gx*cos(t)cos(p) + gy*cos(t)sin(p) - gz*sin(t)
    # grad_p = -gx*sin(t)sin(p) + gy*sin(t)cos(p)
    
    theta = params.reshape(N, 2)[:, 0]
    phi = params.reshape(N, 2)[:, 1]
    
    grad_t = np.zeros(N)
    grad_p = np.zeros(N)
    
    for i in range(N):
        t = theta[i]
        p = phi[i]
        gx, gy, gz = grad_cart[i]
        
        st = np.sin(t)
        ct = np.cos(t)
        sp = np.sin(p)
        cp = np.cos(p)
        
        grad_t[i] = gx * ct * cp + gy * ct * sp - gz * st
        grad_p[i] = -gx * st * sp + gy * st * cp
        
    grad_params = np.zeros_like(params)
    grad_params[0::2] = grad_t
    grad_params[1::2] = grad_p
    
    return obj, grad_params

def construct():
    global START_TIME, TIME_LIMIT
    
    # Generate test points for covering radius estimation
    # Use a dense Fibonacci spiral for test points to approximate the sphere well
    M = 5000
    i_test = np.arange(M) + 0.5
    z_test = 1.0 - 2.0 * i_test / M
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_points = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    best_score = np.inf
    best_params = None
    
    # Initial guess: Fibonacci spiral
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    init_points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    init_params = _cartesian_to_spherical(init_points).flatten()
    
    # Optimization loop with restarts
    num_restarts = 5
    beta = 10.0
    
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Perturb initial parameters for restarts > 0
        if r == 0:
            current_params = init_params.copy()
        else:
            # Random start
            theta = np.random.uniform(0, np.pi, N)
            phi = np.random.uniform(0, 2*np.pi, N)
            current_params = np.column_stack([theta, phi]).flatten()
            
        try:
            res = minimize(
                _objective_and_gradient,
                current_params,
                args=(test_points, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            # Evaluate actual covering radius of the result
            final_points = _spherical_to_cartesian(res.x.reshape(N, 2))
            # Normalize just in case
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            final_points = final_points / norms
            
            # Calculate true covering radius on a slightly denser grid for final score check?
            # The evaluator will do its own check. We just need to return a good array.
            # We can use the smooth objective value as a proxy for ranking, 
            # but let's compute the max distance on the test points for a more accurate local score.
            local_score = _covering_radius(final_points, test_points)
            
            if local_score < best_score:
                best_score = local_score
                best_params = res.x.copy()
                
        except Exception as e:
            # If optimization fails, keep previous best
            pass

    if best_params is None:
        # Fallback to initial
        return init_points
    
    best_points = _spherical_to_cartesian(best_params.reshape(N, 2))
    # Ensure normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
