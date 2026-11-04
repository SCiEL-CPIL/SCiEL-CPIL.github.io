import numpy as np
import time

N = 150

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    def compute_energy(X):
        S = X @ X.T
        mask = ~np.eye(N, dtype=bool)
        vals = 2.0 * (1.0 - S[mask])
        vals = np.maximum(vals, 1e-12)
        return -0.5 * np.sum(np.log(vals))
    
    def is_valid(X):
        if not np.all(np.isfinite(X)):
            return False
        # Check min distance
        # D^2 = 2(1-S)
        S = X @ X.T
        mask = ~np.eye(N, dtype=bool)
        d2 = 2.0 * (1.0 - S[mask])
        if np.min(d2) < 1e-18: # (1e-9)^2
            return False
        return True

    best_E = compute_energy(X)
    best_X = X.copy()
    
    eta = 0.05
    min_eta = 1e-10
    max_iter = 50000
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        S = X @ X.T
        denom = 2.0 * (1.0 - S)
        np.fill_diagonal(denom, 1.0)
        W = 1.0 / denom
        np.fill_diagonal(W, 0.0)
        
        r = W.sum(axis=1)
        
        G = -np.diag(r) @ X + W @ X
        
        dot = np.einsum('ij,ij->i', G, X)
        G_tan = G - np.outer(dot, X)
        
        X_new = X - eta * G_tan
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        if not np.all(np.isfinite(X_new)):
            break
            
        E_new = compute_energy(X_new)
        
        if E_new < best_E and is_valid(X_new):
            best_E = E_new
            best_X = X_new.copy()
            X = X_new
            # Slight decay
            # eta *= 0.9999
        else:
            eta *= 0.8
            if eta < min_eta:
                break
            continue
            
    return best_X
