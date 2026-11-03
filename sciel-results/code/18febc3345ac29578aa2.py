import numpy as np
import time
import math

N = 150

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    def energy(X):
        # Compute pairwise squared distances
        # D2[i, j] = ||x_i - x_j||^2
        XXT = X @ X.T
        D2 = 2.0 - 2.0 * XXT
        # Avoid log(0) and division by zero
        np.fill_diagonal(D2, 1.0)
        # Energy is sum_{i<j} log(1/d) = - sum_{i<j} log(d)
        # We compute sum over all i != j, then divide by 2
        # log(d) = 0.5 * log(D2)
        # Sum_{i != j} log(d_ij) = 0.5 * Sum_{i != j} log(D2_ij)
        # Energy = -0.5 * Sum_{i != j} log(D2_ij)
        # Since diagonal is 1, log(1)=0, we can just sum all
        return -0.5 * np.sum(np.log(D2))

    def grad(X):
        XXT = X @ X.T
        D2 = 2.0 - 2.0 * XXT
        np.fill_diagonal(D2, 1.0)
        W = 1.0 / D2
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1)
        # g_i = sum_j W_ij x_j - x_i S_i
        g = W @ X - X * S[:, np.newaxis]
        # Project to tangent space
        # g_dot_x = sum(g * X, axis=1)
        g_dot_x = np.sum(g * X, axis=1)
        g_tan = g - g_dot_x[:, np.newaxis] * X
        return g_tan

    best_X = X.copy()
    best_E = energy(best_X)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    alpha = 0.01
    max_alpha = 1.0
    min_alpha = 1e-8
    
    while time.time() < deadline:
        g = grad(best_X)
        
        # Backtracking line search
        # We want to minimize E(X - alpha * g_tan)
        # Since g is already tangent, X - alpha * g_tan is close to sphere
        # We normalize after step
        
        accepted = False
        for _ in range(20):
            X_new = best_X - alpha * g
            # Normalize
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            E_new = energy(X_new)
            
            if E_new < best_E:
                best_X = X_new
                best_E = E_new
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            # If we couldn't improve, maybe reduce alpha further or break
            if alpha < min_alpha:
                break
            alpha = min_alpha
        
        # Reset alpha to a reasonable value if we accepted a step
        if accepted:
            alpha = min(alpha * 1.5, max_alpha)
            
    return best_X
