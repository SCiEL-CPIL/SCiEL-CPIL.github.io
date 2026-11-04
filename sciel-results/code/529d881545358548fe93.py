"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    # Use FFT for faster convolution if needed, but N=400 is small.
    # np.convolve is O(N^2) = 160,000 ops, which is fast.
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 110.0  # 110s safety margin for 120s limit
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Helper to run optimization
    def run_optimization(h0, maxiter=5000, timeout=30.0):
        nonlocal best_h, best_val
        start_time = time.time()
        
        # Objective function wrapper
        def objective(x):
            if time.time() > deadline - 5:
                return best_val # Return current best to stop early if possible
            return compute_value(x)
        
        # Use Nelder-Mead as it handles non-smooth functions better
        try:
            res = minimize(objective, h0, method='Nelder-Mead', 
                           options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
        
        # Check time
        return time.time() - start_time

    # Strategy 1: Start from ones and perturb slightly
    # The landscape might be flat near ones, so we need a good kick.
    
    # Try some structured initializations that might break symmetry
    initializations = []
    
    # 1. Random
    np.random.seed(42)
    initializations.append(np.random.randn(N))
    
    # 2. Block: Positive then Negative
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    h_block[N//2:] = -1.0
    # Ensure sum is not zero
    h_block[0] += 0.1 
    initializations.append(h_block)
    
    # 3. Ramp
    h_ramp = np.linspace(1, -1, N)
    h_ramp += 0.1 # Ensure sum != 0
    initializations.append(h_ramp)
    
    # 4. Sparse
    h_sparse = np.zeros(N)
    idx = np.random.choice(N, 10, replace=False)
    h_sparse[idx] = 1.0
    h_sparse[0] = 2.0 # Ensure sum > 0
    initializations.append(h_sparse)

    # Run optimization for each initialization
    for i, h0 in enumerate(initializations):
        if time.time() > deadline - 20:
            break
            
        # Normalize h0 to have sum = N to keep scale consistent
        s = np.sum(h0)
        if abs(s) > 1e-10:
            h0 = h0 / s * N
        else:
            h0 = h0 + 0.1
            s = np.sum(h0)
            h0 = h0 / s * N
            
        elapsed = run_optimization(h0, maxiter=2000)
        
        # If we have time, try a second phase from the best found so far
        if time.time() < deadline - 10 and i == len(initializations) - 1:
             # Refine the best found
             if best_val < 2.0:
                 elapsed = run_optimization(best_h, maxiter=5000)

    return best_h
# EVOLVE-BLOCK-END
