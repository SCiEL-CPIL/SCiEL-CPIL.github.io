import numpy as np
import time
import math

N = 100
TIME_LIMIT = 140.0  # Safety margin from 150s

def _energy_and_gradient(X):
    """
    Compute energy and Riemannian gradient for points X on unit sphere.
    X: (N, 3) array.
    Returns: energy (float), grad (N, 3) array (Riemannian gradient).
    """
    # Compute pairwise squared distances
    # D2[i, j] = ||x_i - x_j||^2 = 2 - 2 x_i . x_j
    dots = X @ X.T
    D2 = 2.0 - 2.0 * dots
    np.fill_diagonal(D2, 1.0)  # Avoid division by zero, we'll zero it out later
    
    # A[i, j] = 1 / D2[i, j]
    # For i==j, we don't want contribution, so set A[i,i]=0
    A = np.zeros_like(D2)
    np.fill_diagonal(D2, 1.0) # Ensure no div by zero for calculation
    A = 1.0 / D2
    np.fill_diagonal(A, 0.0)
    
    # Energy = - sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # log(d_ij) = 0.5 * log(D2_ij)
    # So Energy = -0.5 * 0.5 * sum_{i!=j} log(D2_ij) = -0.25 * sum_{i!=j} log(D2_ij)
    # Sum over all i!=j is sum(A_mask * log(D2_mask))
    # log(D2) is -inf where D2=0, but we set diagonal to 1, so log(1)=0.
    # Off-diagonals are positive.
    log_D2 = np.log(D2)
    # Sum of log(D2) for i!=j
    sum_log_D2 = np.sum(log_D2) # Diagonal is 0
    energy = -0.25 * sum_log_D2
    
    # Gradient calculation
    # g_i = sum_{j != i} (x_i - x_j) / D2_ij
    # g_i = x_i * sum_{j != i} (1/D2_ij) - sum_{j != i} (1/D2_ij) x_j
    S = np.sum(A, axis=1)  # (N,)
    # g = X * S[:, np.newaxis] - A @ X
    g = X * S[:, np.newaxis] - A @ X
    
    # Project to tangent space
    # g_proj = g - (g . x) x
    # g . x = (N-1)/2
    # Let's verify numerically to be safe, or just use the formula.
    # g_dot_x = np.sum(g * X, axis=1)
    # g_proj = g - g_dot_x[:, np.newaxis] * X
    # Using the derived constant (N-1)/2 is faster and exact for unit sphere.
    g_proj = g - 0.5 * (N - 1) * X
    
    return energy, g_proj

def _riemannian_cg(X, max_iter=1000, tol=1e-10, time_start=0.0):
    """
    Riemannian Conjugate Gradient optimization.
    """
    X = X.copy()
    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    energy, g = _energy_and_gradient(X)
    best_energy = energy
    best_X = X.copy()
    
    p = -g  # Initial direction
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-12:
        return best_X, best_energy
        
    eta = 0.1  # Initial step size
    beta = 0.0
    
    for k in range(max_iter):
        # Check time
        if time.time() - time_start > TIME_LIMIT - 5:
            break
            
        # Line search: minimize f(x - eta * p)
        # We use a simple backtracking line search.
        # Since we are on a manifold, we use retraction:
        # X_new = normalize(X - eta * p)
        
        # We want to find eta such that energy decreases.
        # Start with current eta, decrease if energy increases.
        
        # Calculate new X
        X_new = X - eta * p
        # Normalize
        norms_new = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms_new
        
        energy_new, g_new = _energy_and_gradient(X_new)
        
        # Backtracking
        # If energy increased, reduce eta
        while energy_new > energy:
            eta *= 0.5
            if eta < 1e-12:
                break
            X_new = X - eta * p
            norms_new = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms_new
            energy_new, g_new = _energy_and_gradient(X_new)
            
        # Update
        if energy_new < best_energy:
            best_energy = energy_new
            best_X = X_new.copy()
            
        # Convergence check
        g_norm = np.linalg.norm(g_new)
        if g_norm < tol:
            break
            
        # Fletcher-Reeves beta
        g_norm_sq = np.sum(g_new * g_new)
        if g_norm_sq < 1e-20:
            beta = 0.0
        else:
            g_old_norm_sq = np.sum(g * g)
            beta = g_norm_sq / g_old_norm_sq
            # Damping to avoid instability
            beta = min(beta, 0.9)
            
        # New direction
        p = -g_new + beta * p
        
        # Re-normalize direction? Not strictly necessary for CG on sphere with retraction,
        # but helps with numerical stability.
        # Actually, p should be in tangent space.
        # p_proj = p - (p . x) x
        p_dot_x = np.sum(p * X_new, axis=1)
        p = p - p_dot_x[:, np.newaxis] * X_new
        
        X = X_new
        g = g_new
        
        # Adaptive step size
        # If we had to backtrack a lot, reduce eta. If not, maybe increase.
        # Simple heuristic: keep eta roughly stable.
        
    return best_X, best_energy

def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X, best_E = _riemannian_cg(X0, max_iter=2000, time_start=start_time)
    
    # If time remains, try a random restart or perturbation
    # To improve robustness, we can try a few different initializations.
    # But Fibonacci is usually very close to optimal for log energy.
    # Let's try one random restart if there's plenty of time.
    
    if time.time() - start_time < TIME_LIMIT - 20:
        # Random initialization
        np.random.seed(42)
        X_rand = np.random.randn(N, 3)
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        X_rand, E_rand = _riemannian_cg(X_rand, max_iter=1000, time_start=start_time)
        
        if E_rand < best_E:
            best_X = X_rand
            best_E = E_rand
            
    # Final check
    # Ensure points are on sphere
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
