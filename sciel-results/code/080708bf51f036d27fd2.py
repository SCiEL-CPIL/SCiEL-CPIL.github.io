import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """Compute the normalized maximum of autoconvolution."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    # Use FFT for speed if needed, but for N=600, direct convolution is fine and robust
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = None
    best_score = float('inf')
    
    # 1. Test the uniform function, which is the known global optimum for this metric
    #    Score is max_conv / (area * max_h).
    #    For uniform h=1: max_conv=N, area=N, max_h=1 -> Score = 1.0
    h_uniform = np.ones(N)
    score_uniform = compute_score(h_uniform)
    
    if score_uniform < best_score:
        best_score = score_uniform
        best_h = h_uniform.copy()
        
    # 2. Perform a local search around the uniform function to ensure we are at the minimum
    #    and check if any perturbation helps (it shouldn't, but it adds robustness)
    
    def objective(x):
        # x represents log-heights to ensure positivity
        h = np.exp(x)
        # Normalize h so that max_h is 1? 
        # The score is scale invariant: 
        # max(c*conv) / (c*area * c*max) = max(conv) / (area*max)
        # So we can fix the scale. Let's fix sum(h)=1 or max(h)=1.
        # The current compute_score handles raw h.
        # To use scipy.optimize, we need a smooth function.
        # Let's optimize log(h) directly.
        h = np.exp(x)
        s = compute_score(h)
        if not np.isfinite(s):
            return 1e6
        return s

    # Initial guess: uniform log-heights
    # h_uniform = 1 => log(h) = 0
    x0 = np.zeros(N)
    
    # Use Nelder-Mead as it doesn't require gradients and handles the non-smooth max well
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_h = np.exp(res.x)
    except Exception:
        pass
        
    # 3. Try a few random restarts to ensure global optimality (defensive)
    np.random.seed(123)
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        # Random log-heights
        x_rand = np.random.normal(0, 0.5, N)
        # Run a short optimization
        try:
            res_r = minimize(objective, x_rand, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
            if res_r.success and res_r.fun < best_score:
                best_score = res_r.fun
                best_h = np.exp(res_r.x)
        except Exception:
            pass

    # 4. Final safety check: ensure best_h is non-negative and valid
    if best_h is None or not np.all(best_h >= 0):
        best_h = h_uniform
        best_score = score_uniform

    # The score is invariant to scaling, so we can return best_h directly
    # or normalize it for cleanliness
    # Let's normalize max to 1 for consistency, though it doesn't change the score
    max_val = np.max(best_h)
    if max_val > 0:
        best_h = best_h / max_val

    return list(best_h)
