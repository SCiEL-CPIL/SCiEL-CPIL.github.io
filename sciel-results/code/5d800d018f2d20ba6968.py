import numpy as np
import time
import math

N = 100

def fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def energy_and_grad(X):
    """
    Compute energy E = - sum_{i<j} log(||x_i - x_j||)
    and gradient dE/dx_k = - sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    """
    # X is (N, 3)
    # Compute pairwise distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    # Since ||x_i|| = 1, ||x_i - x_j||^2 = 2 - 2 x_i . x_j
    # Let D be the matrix of dot products X @ X.T
    # dist_sq = 2 - 2 D
    # We need to avoid division by zero, so clip dist_sq
    
    D = X @ X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    # Clip to avoid log(0) and division by 0
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Energy = - sum_{i<j} log(dist_ij)
    # Sum over upper triangle
    iu = np.triu_indices(N, k=1)
    E = -np.sum(np.log(dist[iu]))
    
    # Gradient
    # grad_k = - sum_{j != k} (x_k - x_j) / dist_kj^2
    # Let W = 1 / dist_sq
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_k = - sum_j W_kj (x_k - x_j)
    #        = - (sum_j W_kj) x_k + sum_j W_kj x_j
    #        = - W_k_sum x_k + (W @ X)_k
    W_sum = W.sum(axis=1)
    WX = W @ X
    grad = -W_sum[:, np.newaxis] * X + WX
    
    return E, grad

def optimize_sphere(X, time_limit, max_steps=1000):
    """
    Gradient descent on the sphere.
    """
    X = X.copy()
    E, grad = energy_and_grad(X)
    eta = 0.01
    t_start = time.time()
    
    for step in range(max_steps):
        if time.time() - t_start > time_limit:
            break
        
        # Project gradient to tangent space
        # g_tan = grad - (grad . x_k) x_k
        dot = np.sum(grad * X, axis=1)
        g_tan = grad - dot[:, np.newaxis] * X
        
        # Take step
        X_new = X - eta * g_tan
        
        # Normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        # Check energy
        E_new, grad_new = energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            grad = grad_new
            # Increase eta slightly if we made good progress
            eta *= 1.05
            if eta > 0.1: eta = 0.1
        else:
            # Reduce eta
            eta *= 0.5
            if eta < 1e-6:
                break
                
    return X, E

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # Initial configurations
    init_configs = [fibonacci_spiral(N)]
    
    # Add a few random configurations
    for _ in range(3):
        X_rand = np.random.randn(N, 3)
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        init_configs.append(X_rand)
    
    best_X = init_configs[0]
    best_E, _ = energy_and_grad(best_X)
    
    # Optimize each initial config
    for i, X_init in enumerate(init_configs):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        X_opt, E_opt = optimize_sphere(X_init, time_left, max_steps=500)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # Multi-start refinement
    # Continue optimizing from the best solution with perturbations
    while time.time() < deadline - 5:
        # Perturb best_X
        noise = np.random.randn(N, 3) * 0.1
        X_perturbed = best_X + noise
        X_perturbed /= np.linalg.norm(X_perturbed, axis=1, keepdims=True)
        
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        X_opt, E_opt = optimize_sphere(X_perturbed, time_left, max_steps=500)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    return best_X
