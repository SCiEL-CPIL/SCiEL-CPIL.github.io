# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    timeout = 600
    
    # Precompute indices for projections
    x_coords = np.arange(n)
    y_coords = np.arange(n)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    # Projection values
    proj_D = X - Y
    proj_0 = X
    proj_1 = X + Y
    proj_2 = X + 2*Y
    
    # Unique values for each projection
    vals_D = np.unique(proj_D)
    vals_0 = np.unique(proj_0)
    vals_1 = np.unique(proj_1)
    vals_2 = np.unique(proj_2)
    
    # Maps from cell index to bin index
    # We need to pool probabilities
    # Create mapping arrays
    map_D = np.zeros((n, n), dtype=int)
    for i, v in enumerate(vals_D):
        map_D[proj_D == v] = i
        
    map_0 = np.zeros((n, n), dtype=int)
    for i, v in enumerate(vals_0):
        map_0[proj_0 == v] = i
        
    map_1 = np.zeros((n, n), dtype=int)
    for i, v in enumerate(vals_1):
        map_1[proj_1 == v] = i
        
    map_2 = np.zeros((n, n), dtype=int)
    for i, v in enumerate(vals_2):
        map_2[proj_2 == v] = i

    def calculate_score(P):
        P = np.clip(P, 1e-15, None)
        P = P / P.sum()
        
        # Pool probabilities
        prob_D = np.zeros(len(vals_D))
        prob_0 = np.zeros(len(vals_0))
        prob_1 = np.zeros(len(vals_1))
        prob_2 = np.zeros(len(vals_2))
        
        for y in range(n):
            for x in range(n):
                p = P[y, x]
                if p > 0:
                    prob_D[map_D[y, x]] += p
                    prob_0[map_0[y, x]] += p
                    prob_1[map_1[y, x]] += p
                    prob_2[map_2[y, x]] += p
        
        # Entropies
        def entropy(p):
            p = p[p > 0]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))
            
        D = entropy(prob_D)
        M = max(entropy(prob_0), entropy(prob_1), entropy(prob_2))
        
        if M < 1e-9:
            return -1e9
        return D / M

    # Parametric family: P[y,x] proportional to (10 - |x-y|)^(-alpha)
    best_score = -1e9
    best_P = np.ones((n, n)) / (n*n)
    
    # Grid search for alpha
    alphas = np.linspace(0, 3, 61)
    for alpha in alphas:
        if time.time() - start_time > timeout - 10:
            break
        P_raw = np.ones((n, n))
        for y in range(n):
            for x in range(n):
                diff = abs(x - y)
                count = n - diff
                P_raw[y, x] = (1.0 / count) ** alpha
        P = P_raw / P_raw.sum()
        score = calculate_score(P)
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
    # Local optimization using Nelder-Mead on log-weights
    # Initialize with best found
    log_weights = np.log(best_P.flatten() + 1e-15)
    
    def objective(log_w):
        w = np.exp(log_w)
        w = w / w.sum()
        P = w.reshape((n, n))
        return -calculate_score(P)
        
    try:
        res = minimize(objective, log_weights, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() - start_time < timeout - 5:
            w = np.exp(res.x)
            w = w / w.sum()
            P_opt = w.reshape((n, n))
            score_opt = calculate_score(P_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt
    except Exception:
        pass

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    best_P = best_P / best_P.sum()
    
    return best_P

# EVOLVE-BLOCK-END
