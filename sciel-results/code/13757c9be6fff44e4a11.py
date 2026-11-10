import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s for cleanup

    # Precompute indices for pairs
    n = 8
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    
    def objective(x):
        # Check time
        if time.time() > deadline:
            return 1e10
            
        points = x.reshape(n, 2)
        
        # Calculate pairwise distances
        # diff[i, j, k] = points[i, k] - points[j, k]
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Check for zero distances (distinct points)
        if np.any(dists < 1e-10):
            return 1e10
            
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
            
        # Sum of logs of scaled distances
        # log(d_scaled) = log(d) + log(2) - log(D)
        # sum(log(d_scaled)) = sum(log(d)) + 28*log(2) - 28*log(D)
        
        sum_log_d = np.sum(np.log(dists))
        constant = 28.0 * (np.log(2.0) - np.log(D))
        
        total_log = sum_log_d + constant
        
        return -total_log

    # Bounds
    bounds = [(-10.0, 10.0)] * 16
    
    # Initial guess: Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_x = x0
    best_val = objective(x0)
    
    # 1. Differential Evolution (Global Search)
    try:
        if time.time() < deadline - 20.0:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-8, 
                seed=42, 
                init='sobol'
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass

    # 2. Local Refinement with Powell from multiple starts
    # Start 1: Best from DE
    # Start 2-10: Perturbed regular octagons
    starts = [best_x]
    np.random.seed(42)
    for _ in range(10):
        noise = np.random.normal(0, 0.1, 16)
        starts.append(x0 + noise)
    
    for s in starts:
        if time.time() > deadline - 5.0:
            break
        try:
            res2 = minimize(
                objective, 
                s, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass

    best_points = best_x.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
