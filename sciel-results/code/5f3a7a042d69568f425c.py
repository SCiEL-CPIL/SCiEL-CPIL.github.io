import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Constants
    N0 = 1.0
    N4 = 384.0
    N8 = 10321920.0
    N12 = 1962007275520.0
    
    C0_num = 0.5
    C4_num = 1728.0
    C8_num = 87736320.0
    C12_num = 24524904035200.0
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    def objective(c):
        c0, c4, c8 = c
        S = c0 + H4_0 * c4 + H8_0 * c8
        c12 = -S / H12_0
        
        num = C0_num * c0**2 + C4_num * c4**2 + C8_num * c8**2 + C12_num * c12**2
        den = N0 * c0**2 + N4 * c4**2 + N8 * c8**2 + N12 * c12**2
        
        if den == 0:
            return np.inf
        return num / den

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Run optimization
    try:
        start_time = time.time()
        # Use Nelder-Mead for robustness without gradient
        # Bounds are not strictly necessary but help convergence
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > 150: # Safety margin
             # Return best found so far? Nelder-Mead doesn't easily return intermediate best if killed by timeout in Python script execution unless we handle it.
             # But 10000 iterations of simple arithmetic is very fast (<1s).
             pass
             
        if res.success:
            return res.x
        else:
            # If it fails, try BFGS or just return x0 if it's better
            # Check if x0 is better than res.x
            if objective(x0) < objective(res.x):
                return x0
            return res.x
            
    except Exception:
        return x0
