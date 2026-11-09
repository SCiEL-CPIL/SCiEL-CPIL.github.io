import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def construct():
    # Seed coefficients
    c0, c4, c8 = -113 / 100, 1 / 25, 1 / 3240
    
    # Deadline guard
    start_time = time.time()
    time_limit = 200  # Leave 10s margin for 210s limit
    
    def get_c12(c0, c4, c8):
        # Physicists' Hermite polynomials values at 0
        # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=4790016
        # c0*1 + c4*12 + c8*1680 + c12*4790016 = 0
        return -(c0 + 12*c4 + 1680*c8) / 4790016.0

    def objective(vars):
        c0, c4, c8 = vars
        c12 = get_c12(c0, c4, c8)
        coeffs = np.array([c0, c4, c8, c12])
        degrees = np.array([0, 4, 8, 12])
        
        # Calculate norms and moments
        # Integral of e^-x^2 * P(x)^2
        # P(x) = sum c_k H_k(x)
        # H_k are orthogonal with weight e^-x^2
        # <H_k, H_m> = sqrt(pi) 2^k k! delta_km
        # So ||P||^2_weighted = sqrt(pi) * sum c_k^2 2^k k!
        
        # For x^2 moment:
        # <x^2 H_k, H_m>
        # x^2 H_k = (1/2) H_{k+2} + k H_k
        # So x^2 acts as raising/lowering operator.
        # <x^2 H_k, H_m> = 0 unless m=k, k+2, k-2
        # <x^2 H_k, H_k> = k * <H_k, H_k> = k * sqrt(pi) 2^k k!
        # <x^2 H_k, H_{k+2}> = (1/2) * <H_{k+2}, H_{k+2}> = (1/2) sqrt(pi) 2^{k+2} (k+2)!
        # <x^2 H_k, H_{k-2}> = (1/2) * <H_{k-2}, H_{k-2}> = (1/2) sqrt(pi) 2^{k-2} (k-2)!
        
        # We only have even degrees 0, 4, 8, 12.
        # Interactions:
        # H_0 with H_2 (not present)
        # H_4 with H_2, H_4, H_6
        # H_8 with H_6, H_8, H_10
        # H_12 with H_10, H_12, H_14
        
        # Since H_2, H_6, H_10, H_14 are not in our polynomial P,
        # the off-diagonal terms <x^2 H_k, H_m> where m != k are zero
        # BECAUSE the other index (k+2 or k-2) is not in the set {0,4,8,12}.
        # Let's verify:
        # k=0: m=2 (no)
        # k=4: m=2 (no), m=6 (no)
        # k=8: m=6 (no), m=10 (no)
        # k=12: m=10 (no), m=14 (no)
        
        # So <x^2 P, P> = sum c_k^2 <x^2 H_k, H_k>
        # = sum c_k^2 * k * sqrt(pi) 2^k k!
        
        norm_sq = 0.0
        moment_x2 = 0.0
        for c, k in zip(coeffs, degrees):
            if c == 0: continue
            norm_sq += c**2 * (2.0**k) * factorial(k, exact=False)
            moment_x2 += c**2 * k * (2.0**k) * factorial(k, exact=False)
            
        if norm_sq == 0:
            return 1e10
            
        var_x2 = moment_x2 / norm_sq
        # sigma_x^2 sigma_xi^2 = var_x2 * var_x2 = var_x2^2
        # Minimizing var_x2 is equivalent to minimizing the product
        return var_x2

    # Initial guess
    x0 = np.array([c0, c4, c8])
    
    best_result = x0
    best_val = objective(x0)
    
    # Local optimization
    try:
        # Use Nelder-Mead or L-BFGS-B
        # Nelder-Mead is robust for small dims
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000, 'maxfev': 5000})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res.x
            
        # Check time
        if time.time() - start_time < time_limit:
            # Try L-BFGS-B for polish
            res2 = minimize(objective, best_result, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-10})
            if res2.success and res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x

    except Exception:
        pass

    # Return the best coefficients found
    return np.array(best_result)
