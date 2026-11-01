import numpy as np
import time
import math

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient.
    X: (N, 3) array of points on the unit sphere.
    Returns:
        energy: float
        grad: (N, 3) array of Euclidean gradients.
    """
    # Pairwise differences
    # X_i - X_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Distances
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # We compute sum over all i!=j and divide by 2
    # To avoid log(0), we clip distances to a small epsilon
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)
    
    # log(1/d) = -log(d)
    # Sum over i!=j
    log_dists = np.log(safe_dists)
    # Zero out diagonal
    np.fill_diagonal(log_dists, 0.0)
    
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient calculation
    # grad_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    # diff is (x_i - x_j)
    # dists is |x_i - x_j|
    # We need to divide diff by dists^2
    # Note: dists has 1.0 on diagonal, but diff is 0 on diagonal, so 0/1 = 0. Correct.
    # But we used safe_dists for log. For gradient, we should use the actual distances, 
    # but protect against zero.
    # If dist is 0, gradient is undefined. We assume points are distinct.
    # Use safe_dists for stability.
    inv_dists_sq = 1.0 / (safe_dists ** 2)
    # Zero out diagonal
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad = sum_j diff_ij * inv_dists_sq_ij
    # diff is (N, N, 3), inv_dists_sq is (N, N)
    # Multiply each vector by the scalar
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _optimize(X, time_limit, max_iter=10000):
    """
    Optimize the configuration X using Riemannian Gradient Descent.
    """
    X = X.copy()
    # Normalize initial points
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    best_energy, _ = _energy_and_grad(X)
    best_X = X.copy()
    
    start_time = time.time()
    alpha = 1e-3 # Initial learning rate
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        energy, grad = _energy_and_grad(X)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
            
        # Project gradient to tangent space
        # g_perp = g - (g . x) x
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad_perp = grad - dot * X
        
        # Adaptive step size (simple backtracking line search)
        # Try to reduce alpha if energy increases
        # For simplicity, we use a fixed small step or a simple heuristic.
        # Let's try a simple line search: try alpha, if energy increases, reduce alpha.
        # To save time, we do a few steps of backtracking.
        
        # Compute new points
        X_new = X - alpha * grad_perp
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        energy_new, _ = _energy_and_grad(X_new)
        
        # Backtracking line search
        if energy_new > energy:
            # Reduce alpha
            alpha *= 0.5
            # Retry with smaller alpha
            X_new = X - alpha * grad_perp
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            energy_new, _ = _energy_and_grad(X_new)
            
            if energy_new > energy:
                # If still worse, reduce alpha more and skip update for this step? 
                # Or just accept smaller step.
                alpha *= 0.5
                X_new = X - alpha * grad_perp
                norms = np.linalg.norm(X_new, axis=1, keepdims=True)
                X_new = X_new / norms
                energy_new, _ = _energy_and_grad(X_new)
                
                if energy_new > energy:
                    # Give up on this step, reduce alpha further
                    alpha *= 0.5
                    continue
            
        # Accept step
        X = X_new
        # Increase alpha slightly if we made good progress?
        # For stability, keep alpha constant or increase very slowly.
        # alpha = min(alpha * 1.1, 1e-2)
        
    return best_X, best_energy

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Deadline: 150s wall clock, but evaluator kills at 150s. 
    # We should aim to finish well before, say 140s.
    deadline = time.time() + 130.0
    
    best_X = None
    best_energy = float('inf')
    
    # Strategy: Multiple restarts with different initializations
    # 1. Fibonacci sphere
    # 2. Random points
    # 3. Fibonacci + noise
    
    initializers = []
    
    # Fibonacci
    X_fib = _fibonacci_sphere(N)
    initializers.append(X_fib)
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        X_rand = rng.random((N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        initializers.append(X_rand)
        
    # Fibonacci + noise
    for noise_level in [0.1, 0.5]:
        X_noise = X_fib + rng.normal(0, noise_level, (N, 3))
        X_noise = X_noise / np.linalg.norm(X_noise, axis=1, keepdims=True)
        initializers.append(X_noise)
        
    # Run optimization for each initializer
    # Allocate time evenly? Or give more time to the first ones?
    # Let's just run them in sequence, checking time.
    
    for X0 in initializers:
        if time.time() > deadline:
            break
            
        # Estimate remaining time
        remaining = deadline - time.time()
        # Give each optimizer a fraction of the remaining time
        # But we want to ensure at least one completes.
        # Let's give 20s to each, or 1/len(initializers) of remaining?
        # Better: give a fixed chunk, e.g., 10s, or dynamic.
        # If we have 10 initializers, and 130s, that's 13s each.
        time_budget = min(20.0, remaining * 0.8) # Cap at 20s per run, leave 20% buffer
        
        if time_budget < 1.0:
            break
            
        X_opt, energy = _optimize(X0, time_budget, max_iter=5000)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt
            
        # If we found a very good energy, we might stop early?
        # The known best is around -1083.377.
        # If we get close, we can spend more time polishing.
        
    # If we still have time, polish the best result
    if best_X is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        X_opt, energy = _optimize(best_X, remaining, max_iter=10000)
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt
            
    # Final normalization and check
    if best_X is None:
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N)
    else:
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / norms
        
    return best_X
