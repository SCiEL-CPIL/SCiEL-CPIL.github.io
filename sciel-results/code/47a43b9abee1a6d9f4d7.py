import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # We want to maximize the ratio.
    # This is equivalent to minimizing -log(ratio) or minimizing ratio_denominator / ratio_numerator.
    # Let's minimize f(h) = max(h*h) / sum(h^2) subject to sum(h)=1, h>=0.
    # This is non-convex, but we can use SLSQP.
    
    def objective_and_grad(x):
        h = x
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e12, np.zeros_like(h)
        
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-12:
            return 1e12, np.zeros_like(h)
            
        h_conv = np.convolve(h, h, mode='full')
        peak_val = np.max(h_conv)
        
        # Objective: peak_val / l2_sq
        val = peak_val / l2_sq
        
        # Gradient calculation
        # This is tricky because of max.
        # Approximate gradient:
        # d(peak)/dh is the derivative of the max term.
        # d(l2_sq)/dh = 2h
        
        # For SLSQP, we can just provide the function and let it approximate gradient,
        # or provide a sparse gradient.
        # Given N=50, finite difference is okay but slow.
        # Let's try to compute the gradient of the active peak.
        
        # Find indices of peak
        # There can be multiple.
        # We'll use a simple finite difference for the gradient if needed, 
        # but SLSQP has an option to use finite diff.
        
        return val

    def objective_only(x):
        h = x
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-12:
            return 1e12
        h_conv = np.convolve(h, h, mode='full')
        return np.max(h_conv) / l2_sq

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, None) for _ in range(N)]

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Try a few random initializations and the uniform one
    inits = [np.ones(N) / N]
    
    # Add some random sparse inits (Sidon-like)
    np.random.seed(42)
    for _ in range(10):
        # Pick 5-10 random indices
        k = np.random.randint(5, 11)
        idx = np.random.choice(N, k, replace=False)
        h_init = np.zeros(N)
        h_init[idx] = 1.0 / k
        inits.append(h_init)
        
    # Also try a "block" init
    h_block = np.zeros(N)
    h_block[:10] = 0.1
    inits.append(h_block)

    for h0 in inits:
        try:
            res = minimize(
                objective_only, 
                h0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            if res.success:
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                h_opt /= np.sum(h_opt)
                score = _score_raw(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt
        except Exception as e:
            pass

    return list(best_h)
