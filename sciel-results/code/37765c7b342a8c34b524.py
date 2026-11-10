# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    
    def shannon_entropy(p):
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))
        
    def objective(params):
        P = params.reshape(n, n)
        total = np.sum(P)
        if total < 1e-9:
            return 1e9
        P = P / total
        
        # H_{-1}: x - y
        p_minus1 = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                if P[y, x] > 0:
                    p_minus1[x - y + n - 1] += P[y, x]
        H_minus1 = shannon_entropy(p_minus1)
        
        # H_0: x
        p_x = np.sum(P, axis=0)
        H_0 = shannon_entropy(p_x)
        
        # H_1: x + y
        p_xy = np.zeros(2*n - 1)
        for y in range(n):
            for x in range(n):
                if P[y, x] > 0:
                    p_xy[x + y] += P[y, x]
        H_1 = shannon_entropy(p_xy)
        
        # H_2: x + 2y
        p_x2y = np.zeros(3*n - 1)
        for y in range(n):
            for x in range(n):
                if P[y, x] > 0:
                    p_x2y[x + 2*y] += P[y, x]
        H_2 = shannon_entropy(p_x2y)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 1e9
            
        return -H_minus1 / M

    bounds = [(0, 1)] * (n * n)
    
    # Use differential evolution for global optimization
    # maxiter and popsize are tuned for speed
    result = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42, tol=1e-6, polish=False)
    
    if result.success or result.fun < 1e8:
        best_P = result.x.reshape(n, n)
        total = np.sum(best_P)
        if total > 1e-9:
            best_P = best_P / total
            return best_P
            
    # Fallback to uniform if optimization fails
    return np.ones((n, n), dtype=np.float64) / (n * n)

# EVOLVE-BLOCK-END
