import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 14
D = 3

# Precompute pair indices
pair_indices = list(itertools.combinations(range(N), 2))
M = len(pair_indices)
pair_i = np.array([p[0] for p in pair_indices])
pair_j = np.array([p[1] for p in pair_indices])

def _compute_distances_sq(X):
    """
    Compute squared pairwise distances.
    X: (N, D)
    Returns: (M,) array of squared distances.
    """
    diff = X[pair_i] - X[pair_j]
    dist_sq = np.sum(diff**2, axis=1)
    return dist_sq

def _objective_and_grad(x_flat, tau):
    """
    x_flat: (3*N,)
    tau: temperature
    Returns: objective value, gradient
    """
    X = x_flat.reshape(N, D)
    # Center of mass
    X = X - np.mean(X, axis=0, keepdims=True)
    
    dist_sq = _compute_distances_sq(X)
    
    # Soft max
    max_val = np.max(dist_sq)
    exp_max = np.exp(tau * (dist_sq - max_val))
    Z_max = np.sum(exp_max)
    S_max = (1.0 / tau) * (np.log(Z_max) + tau * max_val)
    p_max = exp_max / Z_max
    
    # Soft min
    min_val = np.min(dist_sq)
    exp_min = np.exp(-tau * (dist_sq - min_val))
    Z_min = np.sum(exp_min)
    S_min = -(1.0 / tau) * (np.log(Z_min) - tau * min_val)
    p_min = exp_min / Z_min
    
    if S_min <= 0 or S_max <= 0:
        return 1e10, np.zeros_like(x_flat)
        
    obj = np.log(S_max) - np.log(S_min)
    
    grad_S_max = np.zeros_like(X)
    grad_S_min = np.zeros_like(X)
    
    diff = X[pair_i] - X[pair_j]
    
    np.add.at(grad_S_max, pair_i, 2.0 * p_max[:, np.newaxis] * diff)
    np.add.at(grad_S_max, pair_j, -2.0 * p_max[:, np.newaxis] * diff)
    
    np.add.at(grad_S_min, pair_i, 2.0 * p_min[:, np.newaxis] * diff)
    np.add.at(grad_S_min, pair_j, -2.0 * p_min[:, np.newaxis] * diff)
    
    grad_J = grad_S_max / S_max - grad_S_min / S_min
    
    return obj, grad_J.flatten()

def _exact_score(X):
    dist_sq = _compute_distances_sq(X)
    d_min = np.min(dist_sq)
    d_max = np.max(dist_sq)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _fibonacci_sphere(n, d):
    if d == 3:
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - (i / float(n - 1)) * 2.0
            radius = np.sqrt(1.0 - z * z)
            theta = golden_angle * i
            points[i, 0] = np.cos(theta) * radius
            points[i, 1] = np.sin(theta) * radius
            points[i, 2] = z
        return points
    else:
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = np.inf
    best_X = None
    
    num_restarts = 100 # Higher limit, break on time
    
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0: # Leave 10s for final polish
            break
            
        # Generate initial points
        if restart % 3 == 0:
            # Fibonacci Sphere
            X = _fibonacci_sphere(N, D)
            # Add noise
            X += rng.normal(scale=0.1, size=(N, D))
        elif restart % 3 == 1:
            # Random in unit ball
            X = rng.normal(size=(N, D))
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms * rng.uniform(0.1, 1.0, size=(N, 1))
        else:
            # Random in cube
            X = rng.uniform(-1, 1, size=(N, D))
            
        # Center
        X -= np.mean(X, axis=0, keepdims=True)
        
        # Normalize scale to avoid numerical issues
        # Set mean squared distance to 1
        dist_sq = _compute_distances_sq(X)
        mean_dist_sq = np.mean(dist_sq)
        if mean_dist_sq > 0:
            X = X * np.sqrt(1.0 / mean_dist_sq)
            
        x_flat = X.flatten()
        
        # Progressive tau
        taus = [5, 20, 100, 500, 2000]
        
        for tau in taus:
            if time.time() > deadline - 10.0:
                break
                
            try:
                res = minimize(
                    _objective_and_grad, 
                    x_flat, 
                    args=(tau,), 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-10}
                )
                x_flat = res.x
            except Exception:
                pass
        
        # Polish with Nelder-Mead on exact objective
        def nm_obj(x_flat):
            X = x_flat.reshape(N, D)
            X = X - np.mean(X, axis=0, keepdims=True)
            return _exact_score(X)
            
        if time.time() < deadline - 5.0:
            try:
                res_nm = minimize(
                    nm_obj, 
                    x_flat, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                x_flat = res_nm.x
            except Exception:
                pass
        
        # Final evaluation
        X_final = x_flat.reshape(N, D)
        X_final -= np.mean(X_final, axis=0, keepdims=True)
        
        dist_sq = _compute_distances_sq(X_final)
        if np.min(dist_sq) < 1e-8:
            continue
            
        score = _exact_score(X_final)
        
        if score < best_score:
            best_score = score
            best_X = X_final.copy()
            
    if best_X is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
        
    return best_X
