import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.85

def _log_energy_fast(pts):
    """Vectorized log energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    return np.sum(np.log(1.0 / dists[iu]))

def _log_energy_grad_fast(pts):
    """Vectorized log energy and gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    # Energy
    dists = np.sqrt(dists2)
    iu = np.triu_indices(n, k=1)
    energy = np.sum(np.log(1.0 / dists[iu]))
    
    # Gradient: grad[i] = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # dists2[i,j] = |x_i - x_j|^2
    # We want to sum over j for each i.
    # factor = 1 / dists2
    # grad = sum_j (diff[i,j] / dists2[i,j])
    
    # Avoid division by zero (diagonal is 1, so safe)
    inv_dists2 = 1.0 / dists2
    # Zero out diagonal contributions (they are 0 anyway since diff[i,i]=0)
    np.fill_diagonal(inv_dists2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dists2[i,j]
    # diff shape (n,n,3), inv_dists2 shape (n,n)
    # We need to multiply diff[i,j,:] by inv_dists2[i,j] and sum over j.
    # This is equivalent to (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    
    grad = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_lbfgs(pts, deadline):
    """Run L-BFGS-B optimization."""
    pts = _project_to_sphere(pts)
    x0 = pts.flatten()
    
    def func(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_grad_fast(p)
        return e
        
    def jac(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_grad_fast(p)
        return g.flatten()
        
    options = {
        'maxiter': 2000,
        'gtol': 1e-12,
        'ftol': 1e-15,
        'disp': False,
        'maxfun': 5000,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def _polish(pts, deadline):
    """Steepest descent polish with backtracking line search."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    lr = 0.01
    min_lr = 1e-12
    patience = 0
    
    while time.time() < deadline:
        e, g = _log_energy_grad_fast(pts)
        if e < -1e17: break
        
        # Project gradient to tangent space
        g_t = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14: break
        
        # Backtracking line search
        step = lr * g_t
        new_pts = pts - step # Minimize, so go opposite to gradient
        new_pts = _project_to_sphere(new_pts)
        
        new_e = _log_energy_fast(new_pts)
        
        if new_e < current_energy:
            pts = new_pts
            current_energy = new_e
            patience = 0
            lr = min(lr * 1.1, 1.0) # Increase step slightly
        else:
            lr *= 0.5
            patience += 1
            if patience > 50 or lr < min_lr:
                break
                
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    best_pts = None
    best_energy = float('inf')
    
    # Define multiple starting configurations
    starts = [
        _fibonacci_spiral(N),
        _fibonacci_spiral(N) + np.random.normal(0, 0.1, (N, 3)),
        np.random.normal(0, 1, (N, 3)),
        _fibonacci_spiral(N) + np.random.normal(0, 0.01, (N, 3))
    ]
    
    # Normalize starts
    starts = [_project_to_sphere(s) for s in starts]
    
    # Allocate time for each start
    time_per_start = (deadline - _start_time) / len(starts)
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Optimize this start
        optimized = _optimize_lbfgs(start, deadline)
        e = _log_energy_fast(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized
            
        # If we have time, polish the best one found so far
        if time.time() < deadline - 5.0: # Reserve 5s for final polish
            # Only polish if this is the best so far or if we are near the end
            pass 
            
    # Final polish on the best result
    if best_pts is not None:
        best_pts = _polish(best_pts, deadline)
        best_pts = _project_to_sphere(best_pts)
        
        # Final validity check
        if not np.all(np.isfinite(best_pts)):
            return _fibonacci_spiral(N)
            
        min_d = 1e18
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_d:
                    min_d = d
        if min_d < 1e-9:
            return _fibonacci_spiral(N)
            
    else:
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
