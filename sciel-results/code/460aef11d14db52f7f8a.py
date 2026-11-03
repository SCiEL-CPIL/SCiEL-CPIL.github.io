"""Minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _fast_energy_grad(X):
    """
    Vectorized energy and gradient computation.
    Returns (energy, gradient).
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient: projected to tangent space.
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    d = np.sqrt(d2)
    d_safe = np.where(d < 1e-15, 1e-15, d)
    
    log_d = np.log(d_safe)
    log_d[~mask] = 0.0
    energy = -0.5 * np.sum(log_d)
    
    d2_safe = np.where(d2 < 1e-30, 1e-30, d2)
    inv_d2 = 1.0 / d2_safe
    inv_d2[~mask] = 0.0
    
    weighted_diff = diff * inv_d2[:, :, np.newaxis]
    grad_raw = -0.5 * np.sum(weighted_diff, axis=1)
    
    dot = np.sum(grad_raw * X, axis=1, keepdims=True)
    grad = grad_raw - dot * X
    
    return energy, grad

def _log_energy(X):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return np.inf
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _wrap_energy(x_flat):
    X = x_flat.reshape(N, 3)
    E, _ = _fast_energy_grad(X)
    return E

def _wrap_grad(x_flat):
    X = x_flat.reshape(N, 3)
    _, G = _fast_energy_grad(X)
    return G.flatten()

def construct():
    deadline = time.time() + 140.0
    
    X0 = _fibonacci_spiral()
    
    best_X = X0.copy()
    best_E = _log_energy(best_X)
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if r == 0:
            X_start = X0.copy()
        else:
            noise = np.random.randn(N, 3) * 0.01
            X_start = _normalize(X0 + noise)
        
        res = minimize(
            _wrap_energy,
            X_start.flatten(),
            jac=_wrap_grad,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        X_new = res.x.reshape(N, 3)
        X_new = _normalize(X_new)
        
        E_new = _log_energy(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_X[i] - best_X[j]) < 1e-9:
                best_X[j] += 1e-8 * np.random.randn(3)
                best_X = _normalize(best_X)
                
    return best_X
