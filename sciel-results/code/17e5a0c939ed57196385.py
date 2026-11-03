import numpy as np
import time
import math

N = 150
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def compute_energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient for points on the sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise distances
    # D[i, j] = ||X[i] - X[j]||
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # We sum over i<j. Since dist is symmetric, sum over all i!=j is 2 * sum_{i<j}.
    # So sum_{i<j} log(d) = 0.5 * sum_{i!=j} log(d)
    # log_energy = -0.5 * sum_{i!=j} log(dist[i,j])
    
    # To avoid log(0), we rely on mask.
    # dist is 0 on diagonal.
    log_dist = np.zeros_like(dist)
    log_dist[mask] = np.log(dist[mask])
    
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient of E = - sum_{i<j} log(d_ij)
    # dE/dX_i = - sum_{j != i} d/dX_i log(||X_i - X_j||)
    # = - sum_{j != i} (1/d_ij) * (X_i - X_j)/d_ij
    # = - sum_{j != i} (X_i - X_j)/d_ij^2
    
    # grad[i] = - sum_{j != i} (X[i] - X[j]) / dist[i,j]^2
    # Note: diff[i,j] = X[i] - X[j]
    # So term is -diff[i,j] / dist[i,j]^2
    
    # dist^2
    dist_sq = dist**2
    # Avoid division by zero
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    
    # grad_terms[i,j] = -diff[i,j] / dist_sq_safe[i,j]
    grad_terms = -diff / dist_sq_safe[:, :, np.newaxis]
    
    # Sum over j for each i
    grad = np.sum(grad_terms, axis=1)
    
    return energy, grad

def normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero if a point collapses to 0
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

def optimize(X_init, max_iters=5000, time_limit=100.0):
    """
    Riemannian Gradient Descent with backtracking line search.
    """
    X = X_init.copy()
    E, grad = compute_energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    # Project gradient to tangent space
    # g_tan = grad - (grad . X) X
    grad_dot_X = np.sum(grad * X, axis=1, keepdims=True)
    g_tan = grad - grad_dot_X * X
    
    lr = 0.01
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Backtracking line search
        accepted = False
        current_lr = lr
        while current_lr > 1e-12:
            # Update
            X_new = X - current_lr * g_tan
            X_new = normalize(X_new)
            
            E_new, grad_new = compute_energy_and_grad(X_new)
            
            if E_new < E:
                X = X_new
                E = E_new
                grad = grad_new
                g_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
                accepted = True
                # Increase learning rate slightly if we can
                # lr = min(lr * 1.05, 0.1) 
                break
            else:
                current_lr *= 0.5
        
        if not accepted:
            # If no step accepted, break or reduce lr drastically
            lr *= 0.1
            if lr < 1e-15:
                break
            
        if E < best_E:
            best_E = E
            best_X = X.copy()
            
        # Convergence check
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < 1e-10:
            break
            
    return best_X, best_E

def fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_init(N):
    X = np.random.randn(N, 3)
    return normalize(X)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_X = None
    best_E = float('inf')
    
    # 1. Fibonacci Start
    X_fib = fibonacci_spiral(N)
    X_opt, E_opt = optimize(X_fib, max_iters=5000, time_limit=30.0)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt
        
    # 2. Random Starts
    num_random_starts = 5
    time_per_start = (deadline - time.time()) / (num_random_starts + 1)
    
    for k in range(num_random_starts):
        if time.time() > deadline:
            break
            
        X_rand = random_init(N)
        # Quick pre-optimization to avoid bad local minima
        X_pre, E_pre = optimize(X_rand, max_iters=500, time_limit=time_per_start * 0.5)
        
        if time.time() > deadline:
            break
            
        # Fine optimization
        X_opt, E_opt = optimize(X_pre, max_iters=2000, time_limit=time_per_start * 0.5)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # 3. Perturbation of best
    if time.time() < deadline and best_X is not None:
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline:
                break
            noise = np.random.randn(*best_X.shape) * 0.1
            X_pert = normalize(best_X + noise)
            X_opt, E_opt = optimize(X_pert, max_iters=1000, time_limit=(deadline - time.time()) * 0.9)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
                
    return best_X

# Keep the original function signature
def construct():
    # The above construct is the real one, but we need to match the name.
    # Wait, the prompt says "Keep the function names and signatures of the seed program."
    # The seed program has `def construct():`.
    # So I will just define `construct` as the main entry point.
    pass

# Re-define construct to be the actual implementation
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_X = None
    best_E = float('inf')
    
    # 1. Fibonacci Start
    X_fib = fibonacci_spiral(N)
    X_opt, E_opt = optimize(X_fib, max_iters=5000, time_limit=30.0)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt
        
    # 2. Random Starts
    num_random_starts = 5
    time_per_start = (deadline - time.time()) / (num_random_starts + 1)
    
    for k in range(num_random_starts):
        if time.time() > deadline:
            break
            
        X_rand = random_init(N)
        # Quick pre-optimization to avoid bad local minima
        X_pre, E_pre = optimize(X_rand, max_iters=500, time_limit=time_per_start * 0.5)
        
        if time.time() > deadline:
            break
            
        # Fine optimization
        X_opt, E_opt = optimize(X_pre, max_iters=2000, time_limit=time_per_start * 0.5)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # 3. Perturbation of best
    if time.time() < deadline and best_X is not None:
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline:
                break
            noise = np.random.randn(*best_X.shape) * 0.1
            X_pert = normalize(best_X + noise)
            X_opt, E_opt = optimize(X_pert, max_iters=1000, time_limit=(deadline - time.time()) * 0.9)
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
                
    return best_X
