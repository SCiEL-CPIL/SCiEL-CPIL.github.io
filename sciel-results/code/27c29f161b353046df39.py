import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 140.0 # Safety margin

def construct():
    # Grid
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Basis: Cosines
    # k=0 is DC
    K = 10 # Number of harmonics
    basis = np.zeros((K, N))
    for k in range(K):
        basis[k] = np.cos(2 * np.pi * k * x / 0.5)
        
    # Initial guess: Cosine bump (k=0 and k=1? No, just k=0 is constant. 
    # Cosine bump is cos(pi x / 0.5) which is cos(2 pi x). That is k=1.
    # Wait, cos(2 pi x) is negative in parts.
    # We want a bump that is 0 at edges.
    # cos(pi x / 0.5) = cos(2 pi x).
    # On [-0.25, 0.25], 2 pi x goes from -pi/2 to pi/2. Cos is positive.
    # So k=1 term is the bump.
    # But we need to ensure non-negativity.
    
    # Let's use a parameterization that ensures non-negativity.
    # h = exp(u) is always > 0.
    # But we might want 0.
    # Let's just optimize u and clip.
    
    # Initial u for Cosine Bump
    # h_target = np.cos(2 * np.pi * x)
    # We want to represent this in the basis.
    # It is exactly basis[1].
    # So a[1] = 1, others 0.
    
    a0 = np.zeros(K)
    a0[1] = 1.0 # Cosine bump
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e9 # Penalty
        
        a = params
        h = basis.T @ a
        # Enforce non-negativity
        h = np.maximum(h, 0)
        
        # Convolution
        # Use FFT for speed? N=600 is small.
        c = np.convolve(h, h, mode='full')
        
        max_c = np.max(c)
        sum_h = np.sum(h)
        
        if sum_h == 0:
            return 1e9
            
        return max_c / (sum_h ** 2)
        
    # Optimization
    best_result = a0
    best_val = objective(a0)
    
    # Try L-BFGS-B
    try:
        res = minimize(objective, a0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        if res.success and res.fun < best_val:
            best_result = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Generate final heights
    h = basis.T @ best_result
    h = np.maximum(h, 0)
    
    # Normalize? No, the score is invariant to scaling?
    # M(f) = max(f*f) / (int f)^2.
    # If we scale f by c, max(c^2 f*f) / (c^2 (int f)^2) = M(f).
    # So scaling doesn't matter.
    # But we should return the heights.
    
    return list(h)

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
