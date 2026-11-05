import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leave 50s margin
    
    # Objective: Max of autocorrelation
    def objective(h):
        if time.time() > deadline:
            return np.inf
        auto = np.convolve(h, h)
        return np.max(auto)
    
    def grad(h):
        # Gradient of max(autoconv) is tricky.
        # Subgradient: gradient of the active constraint.
        # autoconv_k = sum_i h_i h_{i+k}
        # grad_k h_j = h_{j+k} + h_{j-k} (with boundary checks)
        # We can approximate or use finite differences if SLSQP doesn't support analytic grad well for max.
        # SLSQP supports analytic gradients.
        # Let's identify the index k* of the max.
        auto = np.convolve(h, h)
        k_max = np.argmax(auto)
        
        # Gradient of (h*h)_{k_max} w.r.t h
        # (h*h)_k = sum_{i} h_i h_{i+k}
        # d/dh_j (h*h)_k = h_{j+k} + h_{j-k}
        # Indices are tricky.
        # Let's use numerical gradient or just let scipy estimate it.
        # Estimating gradient for N=400 might be slow.
        # Let's provide a subgradient.
        
        grad_h = np.zeros(N)
        # The term is sum_{i=0}^{N-1} h_i h_{i+k}
        # Indices of h in the product: i and i+k.
        # If j is in the first factor: i=j, term h_j h_{j+k}. Deriv h_{j+k}.
        # If j is in the second factor: i+k=j => i=j-k. Term h_{j-k} h_j. Deriv h_{j-k}.
        
        # Range of i is 0..N-1.
        # j+k must be in 0..N-1 => j in -k .. N-1-k
        # j-k must be in 0..N-1 => j in k .. N-1+k
        
        # Let's just use scipy's numerical gradient for simplicity and safety, 
        # but it might be slow. 
        # Alternative: use `method='COBYLA'` which doesn't need gradient.
        
        return None

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0})
    
    # Initial guess
    h0 = np.ones(N) / N
    
    # Try SLSQP
    try:
        res = minimize(objective, h0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-10})
        if time.time() > deadline:
            raise TimeoutError
        if res.success:
            h_opt = res.x
        else:
            # Fallback to h0 if optimization fails
            h_opt = h0
    except Exception:
        h_opt = h0

    # Ensure sum is not zero
    if np.sum(h_opt) == 0:
        h_opt = np.ones(N)
        
    return h_opt
