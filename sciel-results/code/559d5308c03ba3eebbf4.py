import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin

    # Initial guess: Seed program roots
    x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Alternative initial guess: Roots of Laguerre L_12
    # from scipy.special import roots_laguerre
    # x_lag, w_lag = roots_laguerre(12)
    # x0 = x_lag

    def objective(params):
        # params are the 12 roots
        if not np.all(params > 0):
            return 1e10
        # Sort to ensure uniqueness/ordering if needed, though poly doesn't care
        roots = np.sort(params)
        
        # Construct polynomial Q(x) = prod (x - r_i)
        # Coefficients in descending power order
        q_coeffs = np.poly(roots)
        
        # Compute Q(x)^4
        # Convolve q_coeffs with itself 3 times to get power 4
        q2 = np.convolve(q_coeffs, q_coeffs)
        q4 = np.convolve(q2, q2)
        
        # Integral of e^-x * Q(x)^4 from 0 to inf
        # If Q(x)^4 = sum c_k x^k, integral is sum c_k * k!
        # q4 is in descending order: c_48, c_47, ..., c_0
        # We need sum_{k=0}^{48} c_k * k!
        # c_k is q4[48-k]
        
        # Calculate factorials up to 48
        # 48! is approx 1.24e61, which fits in float64
        fact = np.ones(49)
        for i in range(1, 49):
            fact[i] = fact[i-1] * i
            
        # c_k corresponds to power k.
        # q4[0] is coeff of x^48.
        # So coeff of x^k is q4[48-k].
        # Sum = sum_{k=0}^{48} q4[48-k] * fact[k]
        
        # Efficient dot product
        # Reverse fact to align with q4
        # q4[i] is coeff of x^(48-i).
        # We want sum_{i=0}^{48} q4[i] * (48-i)!
        # Let's just loop or use array ops
        # powers = np.arange(48, -1, -1)
        # terms = q4 * fact[powers]
        # return np.sum(terms)
        
        # Vectorized:
        powers = np.arange(48, -1, -1)
        return np.sum(q4 * fact[powers])

    # Optimization
    best_x = x0
    best_val = objective(x0)
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline - 10:
        return [float(x) for x in best_x]

    # Try L-BFGS-B with bounds
    bounds = [(1e-6, None)] * 12
    try:
        res2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 1000})
        if res2.success or res2.fun < best_val:
            best_x = res2.x
            best_val = res2.fun
    except Exception:
        pass

    # Ensure positive
    best_x = np.maximum(best_x, 1e-6)
    best_x = np.sort(best_x)
    
    return [float(x) for x in best_x]
