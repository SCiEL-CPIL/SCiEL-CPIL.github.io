import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient.
    X: (N, 3) array of points.
    Returns:
        energy: float
        grad: (N, 3) array of Euclidean gradients.
    """
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Distances
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)
    
    log_dists = np.log(safe_dists)
    np.fill_diagonal(log_dists, 0.0)
    
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient calculation
    # grad_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    inv_dists_sq = 1.0 / (safe_dists ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _objective_and_grad(flat_X):
    """
    Objective function for scipy.optimize.minimize.
    Includes a penalty for points not being on the unit sphere.
    """
    X = flat_X.reshape(N, 3)
    energy, grad = _energy_and_grad(X)
    
    # Penalty term: 0.5 * lambda * sum(||x_i||^2 - 1)^2
    # Gradient of penalty: lambda * (||x_i||^2 - 1) * x_i
    lambda_pen = 100.0
    norms_sq = np.sum(X**2, axis=1)
    penalty = 0.5 * lambda_pen * np.sum((norms_sq - 1.0)**2)
    pen_grad = lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * X
    
    total_energy = energy + penalty
    total_grad = grad + pen_grad
    
    return total_energy, total_grad.flatten()

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_lbfgs(X0, time_limit):
    """
    Optimize using L-BFGS-B with a time limit.
    """
    X0 = X0.copy()
    # Normalize initial points
    norms = np.linalg.norm(X0, axis=1, keepdims=True)
    X0 = X0 / norms
    
    flat_X0 = X0.flatten()
    
    # We use a wrapper to handle the time limit by checking in the callback
    # scipy's minimize doesn't have a direct time limit, so we rely on maxiter
    # and a reasonable guess for iterations, or we can use a custom callback
    # that raises an exception or stops. 
    # However, simple maxiter might not align with time.
    # Let's estimate: L-BFGS is fast. 1000 iterations should be well under 1s usually.
    # To be safe, we run a fixed number of iterations that is likely fast, 
    # or we can use a callback to check time.
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit * 0.9:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_and_grad, 
            flat_X0, 
            method='L-BFGS-B', 
            jac=True, # Since _objective_and_grad returns (f, g)
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except RuntimeError:
        # If time limit exceeded, use the last known best? 
        # Minimize doesn't easily give the last xk on exception.
        # Let's try to catch it differently or just rely on maxiter being small enough.
        # For N=100, L-BFGS is quite fast. 500 iterations is likely < 1-2 seconds.
        # So time limit is likely not hit unless system is very slow.
        # If it is hit, we return the initial guess or best found so far?
        # Actually, if we raise, we lose the result.
        # Better to not raise but check time in a wrapper?
        # Let's stick to maxiter=1000 which is safe for 120s.
        return X0, _energy_and_grad(X0)[0]

    if res.success or res.status == 0:
        X_opt = res.x.reshape(N, 3)
    else:
        # Even if not fully converged, the result might be better
        X_opt = res.x.reshape(N, 3)
        
    # Normalize final result to ensure it's on the sphere
    norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
    X_opt = X_opt / norms
    
    energy, _ = _energy_and_grad(X_opt)
    return X_opt, energy

def construct():
    deadline = time.time() + 130.0
    
    best_X = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Initializers
    initializers = []
    
    # 1. Fibonacci sphere (good structure)
    X_fib = _fibonacci_sphere(N)
    initializers.append(X_fib)
    
    # 2. Random points
    for _ in range(3):
        X_rand = rng.random((N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        initializers.append(X_rand)
        
    # 3. Fibonacci + Noise
    for noise_level in [0.1, 0.5, 1.0]:
        X_noise = X_fib + rng.normal(0, noise_level, (N, 3))
        X_noise = X_noise / np.linalg.norm(X_noise, axis=1, keepdims=True)
        initializers.append(X_noise)
        
    # 4. "Clustered" random - start with points in hemispheres to encourage spreading
    X_hemi = np.zeros((N, 3))
    for i in range(N):
        # Random direction
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        # Bias towards north or south pole
        if i % 2 == 0:
            v[2] = np.abs(v[2]) + 0.5
        else:
            v[2] = -np.abs(v[2]) - 0.5
        v = v / np.linalg.norm(v)
        X_hemi[i] = v
    initializers.append(X_hemi)

    # Run optimization for each initializer
    # We have ~130s. L-BFGS is fast. Let's allow up to 10s per run.
    # If we run 8 initializers, that's 80s. Safe.
    
    for i, X0 in enumerate(initializers):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        # Allocate time per run
        time_budget = min(10.0, remaining * 0.9)
        
        if time_budget < 1.0:
            break
            
        X_opt, energy = _optimize_lbfgs(X0, time_budget)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt
            
    # If we still have time, polish the best result with more iterations
    if best_X is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 10.0
        # Use a larger time budget for the final polish
        X_opt, energy = _optimize_lbfgs(best_X, remaining)
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt
            
    # Final normalization and check
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    else:
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / norms
        
    return best_X
