import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Helper to compute the exact value for comparison
    def get_val(h):
        s = np.sum(h)
        if abs(s) < 1e-9:
            return np.inf
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))

    val_seed = get_val(np.ones(N))
    best_h = np.ones(N)
    best_val = val_seed

    # Soft max temperature
    T = 0.05
    
    def objective(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        # Normalize h to fix scale invariance. This makes the optimization landscape better.
        h_norm = h / s
        conv = np.convolve(h_norm, h_norm)
        # We minimize max(conv) since 2*N is constant and sum(h_norm)=1
        c_max = np.max(conv)
        # Numerical stability: subtract max before exp
        exp_conv = np.exp((conv - c_max) / T)
        soft_m = T * np.log(np.sum(exp_conv)) + c_max
        
        return soft_m

    # Bounds to prevent extreme values
    bounds = [(-10.0, 10.0)] * N
    
    # Try multiple initializations to avoid local minima
    rng = np.random.default_rng(42)
    
    for i in range(10):
        # Start with ones + noise
        # The noise should be small to keep sum large
        if i == 0:
            x0 = np.ones(N)
        else:
            x0 = np.ones(N) + rng.normal(0, 0.1, N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5})
            
            val_res = get_val(res.x)
            
            if val_res < best_val:
                best_val = val_res
                best_h = res.x
                
        except Exception:
            continue

    return best_h
