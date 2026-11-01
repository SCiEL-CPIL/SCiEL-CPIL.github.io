import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def compute_d_sq(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return dists_sq[iu]

def soft_max(x, tau):
    m = np.max(x)
    if m > 700:
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    return tau * np.log(np.sum(np.exp(x / tau)))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_X = None
    best_score = np.inf
    
    # Initializations
    inits = []
    # Random
    for _ in range(10):
        inits.append(np.random.randn(N, D) * 2.0)
        
    # Grid 2x2x4
    grid = np.array(list(np.meshgrid(range(2), range(2), range(4), indexing='ij'))).T.reshape(-1, 3)
    inits.append(grid[:14].astype(float))
    inits.append(grid[:14].astype(float) * 1.1)
    inits.append(grid[:14].astype(float) * 0.9)
    inits.append(grid[:14].astype(float) * 1.2)
    
    # Simple cubic 2x3x3 = 18, take 14
    grid2 = np.array(list(np.meshgrid(range(2), range(3), range(3), indexing='ij'))).T.reshape(-1, 3)
    inits.append(grid2[:14].astype(float))

    for X0 in inits:
        if time.time() > deadline - 15.0:
            break
            
        # Normalize scale: min dist ~ 1
        d_sq = compute_d_sq(X0)
        if np.min(d_sq) > 1e-6:
            X0 = X0 * (1.0 / np.sqrt(np.min(d_sq)))
            
        # Annealing parameters
        tau = 0.5
        lam = 10.0
        
        current_X = X0.flatten()
        
        # Run several steps of annealing
        max_anneal_steps = 15
        
        for step in range(max_anneal_steps):
            if time.time() > deadline - 5.0:
                break
                
            def obj(X_flat):
                X = X_flat.reshape(N, D)
                d_sq = compute_d_sq(X)
                d_sq = np.maximum(d_sq, 1e-8)
                
                s_max = soft_max(d_sq, tau)
                viol = np.maximum(0, 1.0 - d_sq)
                pen = lam * np.sum(viol**2)
                
                return s_max + pen
            
            try:
                res = minimize(
                    obj, 
                    current_X, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success:
                    current_X = res.x
            except Exception:
                break
                
            # Update annealing params
            tau *= 0.8
            lam *= 1.5
            
            # Check actual score
            X_curr = current_X.reshape(N, D)
            d_sq = compute_d_sq(X_curr)
            if np.min(d_sq) > 1e-6:
                score = (np.max(d_sq) / np.min(d_sq))**2
                if score < best_score:
                    best_score = score
                    best_X = X_curr.copy()

    # Final Polish
    if best_X is not None and time.time() < deadline - 2.0:
        tau = 0.05
        lam = 1000.0
        current_X = best_X.flatten()
        
        def obj_polish(X_flat):
            X = X_flat.reshape(N, D)
            d_sq = compute_d_sq(X)
            d_sq = np.maximum(d_sq, 1e-8)
            s_max = soft_max(d_sq, tau)
            viol = np.maximum(0, 1.0 - d_sq)
            pen = lam * np.sum(viol**2)
            return s_max + pen
            
        try:
            res = minimize(
                obj_polish, 
                current_X, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success:
                X_polished = res.x.reshape(N, D)
                d_sq = compute_d_sq(X_polished)
                if np.min(d_sq) > 1e-6:
                    score = (np.max(d_sq) / np.min(d_sq))**2
                    if score < best_score:
                        best_score = score
                        best_X = X_polished
        except Exception:
            pass

    if best_X is None:
        best_X = np.random.randn(N, D)
        
    return best_X
