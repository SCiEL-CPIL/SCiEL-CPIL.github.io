import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140 # 10s margin
    
    # Initial guess
    h = np.full(N, 0.5)
    
    # Objective function
    def objective(h):
        # Compute overlaps O_k for all shifts
        # O_k = sum h_i * (1 - h_{i+k})
        # Vectorized computation
        # h is length N
        # Overlap for shift k is dot(h, roll(h, k)) ? No.
        # O_k = sum h_i (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k} = N/2 - C_k
        # We want to minimize max O_k.
        # Equivalent to minimizing max (N/2 - C_k) = N/2 - min C_k.
        # So we want to maximize min C_k.
        # Let's minimize -min C_k.
        
        # Compute autocorrelation C_k
        # C_k = sum h_i h_{i+k}
        # Using FFT or direct convolution
        # For N=100, direct is fine.
        
        C = np.zeros(N)
        for k in range(N):
            C[k] = np.dot(h, np.roll(h, k))
        
        # Min C_k is the bottleneck
        # Soft min: -1/alpha * log(sum exp(-alpha C_k))
        # We want to maximize min C_k, so minimize -min C_k.
        # Soft max of C_k? No.
        # We want to push all C_k up.
        # Minimize max (N/2 - C_k).
        # Let g_k = N/2 - C_k.
        # Minimize max g_k.
        # Soft max of g_k.
        
        alpha = 10.0
        g = N/2.0 - C
        # Soft max
        max_g = np.max(g)
        # Shift for numerical stability
        g_shifted = g - max_g
        soft_max = max_g + (1/alpha) * np.log(np.sum(np.exp(alpha * g_shifted)))
        
        return soft_max

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0},
        {'type': 'ineq', 'fun': lambda h: h}, # h >= 0
        {'type': 'ineq', 'fun': lambda h: 1 - h} # h <= 1
    ]
    
    # Bounds
    bounds = [(0, 1) for _ in range(N)]
    
    # Optimization
    best_h = h
    best_score = objective(h)
    
    try:
        result = minimize(
            objective, 
            h, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if time.time() < deadline:
            if result.fun < best_score:
                best_h = result.x
                best_score = result.fun
                
    except Exception:
        pass
        
    # Ensure constraints are met strictly
    best_h = np.clip(best_h, 0, 1)
    # Adjust sum to be exactly N/2
    diff = N/2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
