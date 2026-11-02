import numpy as np
import time
from scipy.optimize import minimize
import math

N = 130

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_gradient(X, Y, beta, penalty_weight=100.0):
    """
    Computes the gradient of the soft-max covering radius objective + sphere penalty.
    X: (N, 3) array of points on the sphere.
    Y: (M, 3) array of test points on the sphere.
    beta: temperature parameter for soft-max.
    """
    N_pts = X.shape[0]
    M_pts = Y.shape[0]
    
    # Normalize X to ensure they are on the sphere during gradient calculation
    # (L-BFGS-B might drift slightly, but we penalize it)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Compute dot products: (M, N)
    # Y @ X_norm.T
    dots = Y @ X_norm.T
    
    # Clip for numerical stability in arccos
    dots = np.clip(dots, -1.0, 1.0)
    
    # Find nearest neighbor index for each test point y
    # Max dot product corresponds to min angular distance
    nn_idx = np.argmax(dots, axis=1) # (M,)
    
    # Get the dot products for the nearest neighbors
    dots_nn = dots[np.arange(M_pts), nn_idx] # (M,)
    
    # Angular distances
    angles = np.arccos(dots_nn) # (M,)
    
    # Soft-max weights
    # J = (1/beta) * log(sum(exp(beta * angles)))
    # grad J = sum(w_y * grad angle_y)
    # w_y = exp(beta * angle_y) / sum(exp(beta * angle_y))
    
    # Numerical stability for exp
    max_angle = np.max(angles)
    exp_angles = np.exp(beta * (angles - max_angle))
    sum_exp = np.sum(exp_angles)
    weights = exp_angles / sum_exp # (M,)
    
    # Gradient of angle with respect to x_j
    # d/dx arccos(y . x) = -(y - (y.x)x) / sqrt(1 - (y.x)^2)
    # This is only non-zero for j = nn_idx(y)
    
    grads = np.zeros_like(X)
    
    # We need to accumulate gradients for each x_j
    # For each y, the gradient is applied to X[nn_idx[y]]
    
    # Vectorized calculation of the gradient vector for each y
    # y: (M, 3), x_nn: (M, 3)
    x_nn = X_norm[nn_idx]
    
    # y_dot_x = dots_nn (M,)
    # proj_y_on_x = y_dot_x * x_nn
    # diff = y - proj_y_on_x
    # norm_diff = sqrt(1 - y_dot_x^2)
    
    y_dot_x = dots_nn
    diff = Y - y_dot_x[:, np.newaxis] * x_nn
    norm_diff = np.sqrt(np.clip(1.0 - y_dot_x**2, 1e-12, 1.0))
    
    # grad_angle = -diff / norm_diff
    grad_angles = -diff / norm_diff[:, np.newaxis] # (M, 3)
    
    # Weighted gradients
    weighted_grads = weights[:, np.newaxis] * grad_angles # (M, 3)
    
    # Scatter-add to grads
    # np.add.at is slow, but M is not huge (e.g., 5000). 
    # Alternatively, use bincount or loop if M is small.
    # For M=5000, N=130, a loop in Python is too slow. 
    # We can use np.add.at or a matrix multiplication trick if indices were unique, but they are not.
    # np.add.at is acceptable for M=5000.
    np.add.at(grads, nn_idx, weighted_grads)
    
    # Add sphere constraint penalty gradient
    # Penalty P = penalty_weight * sum(|x_i|^2 - 1)^2
    # dP/dx_i = 4 * penalty_weight * (|x_i|^2 - 1) * x_i
    norms_sq = np.sum(X**2, axis=1)
    penalty_grad = 4.0 * penalty_weight * (norms_sq - 1.0)[:, np.newaxis] * X
    
    return grads + penalty_grad

def _objective(X, Y, beta, penalty_weight=100.0):
    """
    Computes the soft-max covering radius + sphere penalty.
    """
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    dots = Y @ X_norm.T
    dots = np.clip(dots, -1.0, 1.0)
    nn_idx = np.argmax(dots, axis=1)
    dots_nn = dots[np.arange(len(Y)), nn_idx]
    angles = np.arccos(dots_nn)
    
    max_angle = np.max(angles)
    exp_angles = np.exp(beta * (angles - max_angle))
    soft_max = (1.0 / beta) * np.log(np.sum(exp_angles)) + max_angle
    
    norms_sq = np.sum(X**2, axis=1)
    penalty = penalty_weight * np.sum((norms_sq - 1.0)**2)
    
    return soft_max + penalty

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Generate test points
    # Use a dense Fibonacci spiral for test points
    M_test = 20000
    Y = _fibonacci_sphere(M_test)
    
    # Initial guess
    X = _fibonacci_sphere(N)
    
    # Parameters
    beta = 50.0
    penalty_weight = 100.0
    
    def obj_func(x_flat):
        X_curr = x_flat.reshape(N, 3)
        return _objective(X_curr, Y, beta, penalty_weight)
        
    def jac_func(x_flat):
        X_curr = x_flat.reshape(N, 3)
        return _compute_gradient(X_curr, Y, beta, penalty_weight).flatten()
        
    x0 = X.flatten()
    
    best_x = x0.copy()
    best_val = obj_func(x0)
    
    # Optimization loop
    # We can run multiple restarts or just one long L-BFGS-B
    # L-BFGS-B is efficient.
    
    try:
        res = minimize(
            obj_func, 
            x0, 
            jac=jac_func, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100000}
        )
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Try a second round with higher beta to sharpen
            beta2 = 100.0
            x0_2 = best_x.copy()
            
            def obj_func2(x_flat):
                X_curr = x_flat.reshape(N, 3)
                return _objective(X_curr, Y, beta2, penalty_weight)
                
            def jac_func2(x_flat):
                X_curr = x_flat.reshape(N, 3)
                return _compute_gradient(X_curr, Y, beta2, penalty_weight).flatten()
                
            res2 = minimize(
                obj_func2, 
                x0_2, 
                jac=jac_func2, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
            )
            
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
                
    except Exception:
        pass
        
    # Final normalization
    X_final = best_x.reshape(N, 3)
    X_final = X_final / np.linalg.norm(X_final, axis=1, keepdims=True)
    
    return X_final

# EVOLVE-BLOCK-END
