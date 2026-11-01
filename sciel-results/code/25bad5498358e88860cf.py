import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(X, penalty_coeff=0.0):
    """
    Compute logarithmic energy and its gradient.
    X: (N, 3) array of points.
    Returns energy (scalar) and grad (N, 3) array.
    """
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Squared distances
    d2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for self-distances
    np.fill_diagonal(d2, 1.0)
    
    # Weights 1/d2
    w = 1.0 / d2
    np.fill_diagonal(w, 0.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Note: log(1/d) = -log(d). 
    # Sum over i<j is half of sum over i!=j.
    # E = -0.5 * sum_{i!=j} log(sqrt(d2_ij)) = -0.25 * sum_{i!=j} log(d2_ij)
    # Let's stick to the definition: sum_{i<j} log(1/d_ij)
    # We can compute the full sum and divide by 2.
    # log(1/d) = -0.5 * log(d2)
    # Sum_{i!=j} -0.5 * log(d2_ij) / 2 = -0.25 * sum_{i!=j} log(d2_ij)
    
    # Using log(d2) is numerically more stable than log(d)
    log_d2 = np.log(d2)
    np.fill_diagonal(log_d2, 0.0)
    
    energy = -0.25 * np.sum(log_d2)
    
    # Gradient of energy
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / d2_ij
    # This is - sum_j (x_i - x_j) * w_ij
    grad = -np.sum(diff * w[:, :, np.newaxis], axis=1)
    
    # Add penalty gradient if penalty_coeff > 0
    if penalty_coeff > 0:
        norms2 = np.sum(X**2, axis=1)
        # Penalty term: 0.5 * penalty_coeff * sum (||x_i||^2 - 1)^2
        # Grad: penalty_coeff * (||x_i||^2 - 1) * x_i
        grad += penalty_coeff * (norms2 - 1.0)[:, np.newaxis] * X
        # Penalty energy
        energy += 0.5 * penalty_coeff * np.sum((norms2 - 1.0)**2)
        
    return energy, grad

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin for safety
    
    # Initial guess: Fibonacci spiral
    x0 = _fibonacci_sphere(N)
    
    # We will use L-BFGS-B on 3D coordinates with a penalty for norm constraint.
    # Start with a moderate penalty to guide the optimizer.
    penalty_coeff = 1000.0
    
    best_x = x0.copy()
    best_e, _ = _energy_and_grad(x0, penalty_coeff=0.0) # True energy without penalty for tracking
    
    # Callback to check time
    def callback(xk):
        nonlocal best_x, best_e
        if time.time() > deadline:
            # We can't easily break out of minimize, but we can record best
            # Actually, minimize will continue. We'll just let it run and check time after.
            pass
        # Update best if energy (without penalty) is lower
        e_true, _ = _energy_and_grad(xk, penalty_coeff=0.0)
        if e_true < best_e:
            best_e = e_true
            best_x = xk.copy()
        
    # Run L-BFGS-B
    try:
        res = minimize(
            lambda x: _energy_and_grad(x.reshape(N, 3), penalty_coeff)[0],
            x0.flatten(),
            jac=lambda x: _energy_and_grad(x.reshape(N, 3), penalty_coeff)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        # Update best from result
        x_res = res.x.reshape(N, 3)
        e_res, _ = _energy_and_grad(x_res, penalty_coeff=0.0)
        if e_res < best_e:
            best_e = e_res
            best_x = x_res.copy()
    except Exception as e:
        # If minimize fails, we still have best_x from callback or initial
        pass
    
    # Check time
    if time.time() < deadline - 10.0:
        # Polish phase: Projected Gradient Descent
        x_curr = best_x.copy()
        # Adaptive step size
        step = 1e-4
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            e_curr, g_curr = _energy_and_grad(x_curr, penalty_coeff=0.0)
            
            # Project gradient to tangent space
            # g_tan = g - (g . x) x
            dot = np.sum(g_curr * x_curr, axis=1)
            g_tan = g_curr - dot[:, np.newaxis] * x_curr
            
            # Line search or fixed step
            # Simple backtracking line search
            alpha = step
            improved = False
            for _ in range(10):
                x_new = x_curr - alpha * g_tan
                # Normalize
                x_new = x_new / np.linalg.norm(x_new, axis=1)[:, np.newaxis]
                
                e_new, _ = _energy_and_grad(x_new, penalty_coeff=0.0)
                if e_new < e_curr - 1e-10 * alpha * np.sum(g_tan**2):
                    improved = True
                    break
                alpha *= 0.5
            
            if improved:
                x_curr = x_new
                if e_new < best_e:
                    best_e = e_new
                    best_x = x_curr.copy()
            else:
                # Reduce step size
                step *= 0.5
                if step < 1e-10:
                    break
        # Ensure best_x is updated from x_curr if it was improved in the last step
        e_final, _ = _energy_and_grad(x_curr, penalty_coeff=0.0)
        if e_final < best_e:
            best_e = e_final
            best_x = x_curr.copy()

    # Final normalization to ensure unit sphere
    best_x = best_x / np.linalg.norm(best_x, axis=1)[:, np.newaxis]
    
    return best_x
