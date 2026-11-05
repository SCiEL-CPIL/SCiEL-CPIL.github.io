"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Approach: Convex Optimization
    # We minimize the maximum autocorrelation peak for a fixed sum.
    # The value is scale invariant, so we fix sum(h) = 1.
    # We minimize max(conv(h,h)) using a log-sum-exp approximation.
    
    def objective(h):
        conv = np.convolve(h, h)
        lam = 50.0
        c_max = np.max(conv)
        # Stable log-sum-exp
        shifted = lam * (conv - c_max)
        shifted = np.clip(shifted, -500, 500)
        log_sum = c_max * lam + np.log(np.sum(np.exp(shifted)))
        return log_sum / lam

    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    h0 = np.ones(N) / N
    
    try:
        res = minimize(objective, h0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 100, 'ftol': 1e-8})
        if res.success and np.isfinite(res.x).all() and abs(np.sum(res.x) - 1.0) < 1e-5:
            return res.x
    except Exception:
        pass
        
    # Fallback to seed if optimization fails or times out
    return np.ones(N)
