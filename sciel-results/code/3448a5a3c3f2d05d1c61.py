import numpy as np
import time
import math
from scipy.optimize import minimize_scalar

N = 150
TIME_LIMIT = 140.0 # Safety margin for 150s kill

def _energy_and_grad(X):
    """
    Compute logarithmic energy and Riemannian gradient.
    X: (N, 3) array of unit vectors.
    Energy: sum_{i<j} log(1/d_ij)
    Grad: Tangent space gradient.
    """
    # Pairwise distances
    # D[i,j] = ||X[i] - X[j]||
    # Use broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} -log(d_ij)
    # We need to sum only i<j
    # Mask for i<j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    E = -np.sum(np.log(d[mask]))
    
    # Gradient
    # grad_E_i = sum_{j != i} (X_i - X_j) / d_ij^2
    # Note: d_ij^2 = d2
    # Term (X_i - X_j) / d2_ij
    # diff is (N, N, 3) where diff[i,j] = X_i - X_j
    # We need to sum over j for each i
    # Weight matrix W[i,j] = 1/d2[i,j] for i!=j, 0 otherwise
    W = 1.0 / d2
    np.fill_diagonal(W, 0.0)
    
    # grad_E = W @ X ? No.
    # grad_E_i = sum_j W[i,j] * (X_i - X_j)
    # = sum_j W[i,j] X_i - sum_j W[i,j] X_j
    # = X_i * (sum_j W[i,j]) - sum_j (W[i,j] X_j)
    
    row_sums = np.sum(W, axis=1)
    # sum_j W[i,j] X_j is W @ X
    WX = W @ X
    
    grad_eucl = row_sums[:, np.newaxis] * X - WX
    
    # Project to tangent space
    # g_i = grad_eucl_i - (grad_eucl_i . X_i) X_i
    proj = np.sum(grad_eucl * X, axis=1)
    grad_r = grad_eucl - proj[:, np.newaxis] * X
    
    return E, grad_r

def _retract(X, P, alpha):
    """
    Retraction: move from X along tangent vector P by alpha, then normalize.
    """
    X_new = X + alpha * P
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return X_new / norms

def _line_search(X, P, E_curr, alpha_init=0.1):
    """
    Backtracking line search for Riemannian Gradient Descent.
    Minimize E(retract(X, alpha P))
    """
    alpha = alpha_init
    rho = 0.1 # Armijo parameter
    c = 1e-4 # Slope parameter
    # Gradient at X
    # We need the directional derivative
    # dE/dalpha at alpha=0 is <grad_r, P>
    # Since P is usually -grad_r (for GD), this is -||grad_r||^2
    # For general P, we can just evaluate energy
    
    # Simple backtracking
    E_prev = E_curr
    for _ in range(50):
        X_new = _retract(X, P, alpha)
        E_new, _ = _energy_and_grad(X_new)
        if E_new < E_prev - c * alpha * np.dot(P.flatten(), _energy_and_grad(X)[1].flatten()): # This is inefficient, re-computes grad
             # Better: just check sufficient decrease
             pass
        if E_new < E_curr:
            return alpha, X_new, E_new
        alpha *= 0.5
    return alpha, X, E_curr

def _optimize(X_init, time_budget):
    """
    Run Riemannian Gradient Descent with adaptive step size.
    """
    X = X_init.copy()
    E_curr, G_curr = _energy_and_grad(X)
    
    # Normalize initial points just in case
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    E_curr, G_curr = _energy_and_grad(X)
    
    alpha = 0.01
    start_time = time.time()
    
    for it in range(10000):
        if time.time() - start_time > time_budget:
            break
            
        # P = -G_curr
        P = -G_curr
        
        # Line search
        # We want to find alpha that minimizes E(retract(X, alpha P))
        # Use a simple scalar minimization around current alpha
        # Or backtracking
        
        best_alpha = alpha
        best_E = E_curr
        best_X = X
        
        # Try a few alpha values
        for a in [alpha * 0.5, alpha, alpha * 1.5, alpha * 2.0]:
            if a <= 0: continue
            X_trial = _retract(X, P, a)
            E_trial, G_trial = _energy_and_grad(X_trial)
            if E_trial < best_E:
                best_E = E_trial
                best_alpha = a
                best_X = X_trial
                best_G = G_trial
        
        if best_E >= E_curr:
            # No improvement, reduce alpha
            alpha *= 0.5
            if alpha < 1e-12:
                break
            continue
            
        # Update
        X = best_X
        E_curr = best_E
        G_curr = best_G
        
        # Convergence check
        if np.linalg.norm(G_curr) < 1e-10:
            break
            
        # Update alpha strategy
        # If we used the best alpha, maybe increase slightly next time?
        # Keep it simple: alpha = best_alpha * 0.8 (shrink a bit to be safe)
        alpha = best_alpha * 0.9
        
    return X, E_curr

def construct():
    start_time = time.time()
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Random
    rng = np.random.default_rng(42)
    x_rand = rng.normal(size=(N, 3))
    x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
    inits.append(x_rand)
    
    # 3. Jittered Fibonacci
    x_jit = x_fib + rng.normal(scale=0.05, size=(N, 3))
    x_jit /= np.linalg.norm(x_jit, axis=1, keepdims=True)
    inits.append(x_jit)
    
    best_X = None
    best_E = float('inf')
    
    # Allocate time to each initialization
    # Total time ~ 140s
    # Let's give 40s to Fibonacci, 40s to Random, 40s to Jittered
    # And keep the best
    
    for idx, X_init in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        budget = min(40.0, TIME_LIMIT - (time.time() - start_time) - 5)
        if budget <= 0:
            continue
            
        X_opt, E_opt = _optimize(X_init, budget)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # Final polish?
    # If time remains, do a high-precision run on the best X
    if best_X is not None and time.time() - start_time < TIME_LIMIT - 5:
        remaining = TIME_LIMIT - (time.time() - start_time) - 2
        if remaining > 1.0:
             X_polished, E_polished = _optimize(best_X, remaining)
             if E_polished < best_E:
                 best_X = X_polished
                 best_E = E_polished

    if best_X is None:
        return x_fib
        
    # Ensure output is valid
    # Normalize again
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
