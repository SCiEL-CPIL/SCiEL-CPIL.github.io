import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 180.0  # seconds
    
    # Hermite polynomials values at 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # Constraint: c0*1 + c4*(-3) + c8*105 + c12*(-10395) = 0
    # c12 = (c0 - 3*c4 + 105*c8) / 10395
    
    # Variances for psi_n: v_n = n + 0.5
    v0 = 0.5
    v4 = 4.5
    v8 = 8.5
    v12 = 12.5
    
    def objective(params):
        c0, c4, c8 = params
        c12 = (c0 - 3*c4 + 105*c8) / 10395.0
        
        num = c0**2 * v0 + c4**2 * v4 + c8**2 * v8 + c12**2 * v12
        den = c0**2 + c4**2 + c8**2 + c12**2
        
        if den < 1e-10:
            return 1e10
            
        ratio = num / den
        return ratio**2
    
    # Initial guess from seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Try multiple starting points to avoid local minima
    best_result = x0
    best_score = objective(x0)
    
    # Generate some random starts
    np.random.seed(42)
    for i in range(50):
        # Random starts in a reasonable range
        start = np.random.uniform(-2, 2, 3)
        try:
            res = opt.minimize(objective, start, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
            
    return np.array(best_result)
