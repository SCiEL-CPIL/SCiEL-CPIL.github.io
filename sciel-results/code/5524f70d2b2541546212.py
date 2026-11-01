import numpy as np
import time
from scipy.optimize import minimize

N = 130
M = 2000  # Number of test points for covering radius estimation
BETA = 50.0  # Soft-max temperature
LAMBDA_NORM = 100.0  # Penalty for deviation from unit sphere

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on a unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for evaluation overhead
    
    # Precompute test points
    test_points = _fibonacci_sphere(M, offset=0.5)
    
    # Initial configuration: Fibonacci spiral
    x0 = _fibonacci_sphere(N, offset=0.5)
    
    def objective(params):
        points = params.reshape(N, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-12] = 1.0
        points = points / norms
        
        # Calculate squared distances from test points to all configuration points
        # d^2 = 2 - 2 * (p . x)
        dots = test_points @ points.T  # Shape (M, N)
        d_sq = 2.0 - 2.0 * dots
        
        # Find minimum squared distance for each test point
        min_d_sq = np.min(d_sq, axis=1)  # Shape (M,)
        
        # Soft-max approximation of the maximum squared distance
        # E = (1/beta) * log(sum(exp(beta * min_d_sq)))
        scaled = BETA * min_d_sq
        max_scaled = np.max(scaled)
        # Subtract max to prevent overflow
        exp_sum = np.sum(np.exp(scaled - max_scaled))
        e_cover = (1.0 / BETA) * (np.log(exp_sum) + max_scaled)
        
        # Penalty for not being on the unit sphere
        # We normalized, so this should be 0, but gradients might push off.
        # Actually, if we normalize inside, the penalty is always 0 for the value,
        # but the gradient calculation in scipy might get confused if we don't penalize the raw input.
        # It's better to penalize the raw input deviation from unit length.
        raw_norms_sq = np.sum(points**2, axis=1)
        e_norm = np.sum((raw_norms_sq - 1.0)**2)
        
        return e_cover + LAMBDA_NORM * e_norm

    def jac(params):
        # Finite difference approximation if analytical is too complex or risky
        # L-BFGS-B can use finite differences, but it's slow.
        # Let's try to let scipy use finite differences first, or implement a simple one.
        # For now, rely on scipy's finite difference or provide a simple one.
        # Actually, providing an analytical Jacobian is complex due to the min and soft-max.
        # Let's use a small step size finite difference if needed, or just let scipy do it.
        # scipy.optimize.minimize with L-BFGS-B uses finite differences by default if jac is not provided.
        # This is acceptable for N=390 parameters.
        pass

    best_x = x0.flatten()
    best_score = objective(best_x)
    
    # Optimization Loop with Restarts
    # We will try to optimize from the initial state.
    # If time permits, we might try a random restart.
    
    try:
        res = minimize(
            objective, 
            best_x, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_score:
            best_x = res.x
            best_score = res.fun
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            # Try a second optimization from a perturbed state
            # Perturb best_x
            noise = np.random.normal(0, 0.1, size=N*3)
            x2 = best_x + noise
            res2 = minimize(
                objective, 
                x2, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res2.success and res2.fun < best_score:
                best_x = res2.x
                best_score = res2.fun

    except Exception:
        pass
        
    # Final normalization
    final_points = best_x.reshape(N, 3)
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    final_points = final_points / norms
    
    return final_points
