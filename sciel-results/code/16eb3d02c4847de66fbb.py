import numpy as np
import scipy.optimize as so
import time
from math import gamma, factorial

def construct():
    start_time = time.time()
    deadline = start_time + 580.0
    
    m = 12
    n = 1
    num_roots = 12
    
    # Bounds for roots
    # Roots should be positive.
    # Laguerre roots for L_12^1 are roughly in [0, 30].
    # Double roots might spread them out or cluster them.
    # Let's use a wide range.
    bounds = [(0.01, 100.0) for _ in range(num_roots)]
    
    def objective(roots):
        # roots: array of 12 values
        # Compute coefficients of P(x) = prod (x - r_i)^2
        # This is equivalent to prod (x - r_i) squared.
        # Let Q(x) = prod (x - r_i). Then P(x) = Q(x)^2.
        
        # Compute Q(x) coefficients
        # Start with [1]
        q_coeffs = np.array([1.0])
        for r in roots:
            # Multiply by (x - r)
            # [c_k, ..., c_0] * (x - r)
            # new[0] = old[0] * 1
            # new[k] = old[k-1] - r * old[k]
            # new[-1] = -r * old[-1]
            new_coeffs = np.zeros(len(q_coeffs) + 1)
            new_coeffs[1:] = q_coeffs
            new_coeffs[:-1] -= r * q_coeffs
            q_coeffs = new_coeffs
            
        # P(x) = Q(x)^2
        # p_coeffs = np.convolve(q_coeffs, q_coeffs)
        # Note: numpy convolve gives coefficients in descending order? 
        # np.convolve([1, -r], [1, -r]) -> [1, -2r, r^2] which is x^2 - 2rx + r^2. Correct.
        p_coeffs = np.convolve(q_coeffs, q_coeffs)
        
        # Compute L2 norm squared: sum_{j,k} p_j p_k Gamma(j+k+n+1)
        # p_coeffs[j] is coefficient of x^(24-j) if we assume descending?
        # Let's check np.convolve order.
        # np.convolve([1, 2], [3, 4]) -> [3, 10, 8].
        # (x+2)(3x+4) = 3x^2 + 10x + 8.
        # So p_coeffs[0] is coeff of x^24, p_coeffs[-1] is coeff of x^0.
        
        norm_sq = 0.0
        # To avoid O(N^2) in Python, use vectorization or loop. N=25, so 25*25=625 ops. Loop is fine.
        for j in range(len(p_coeffs)):
            pj = p_coeffs[j]
            if pj == 0: continue
            for k in range(len(p_coeffs)):
                pk = p_coeffs[k]
                if pk == 0: continue
                # Power of x in term j is 24-j. Power in term k is 24-k.
                # Total power is 48 - j - k.
                # Integral of x^(48-j-k) * e^-x * x^1 dx = Gamma(48-j-k+1+1) = Gamma(50-j-k)
                power = 48 - j - k
                if power < 0: continue
                norm_sq += pj * pk * gamma(power + n + 1)
                
        return norm_sq

    # Global Optimization
    best_result = None
    best_val = np.inf
    
    # Use Differential Evolution
    try:
        res = so.differential_evolution(objective, bounds, seed=42, maxiter=1000, tol=1e-8, polish=False, workers=1)
        if time.time() > deadline - 60:
            # Not enough time for polish, return DE result
            return list(res.x)
        
        best_val = res.fun
        best_result = res.x
        
        # Local Polish
        # Use L-BFGS-B with bounds
        res2 = so.minimize(objective, best_result, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        if res2.fun < best_val:
            best_result = res2.x
            best_val = res2.fun
            
    except Exception as e:
        # Fallback to seed if optimization fails
        return [4.0 * (i + 1) for i in range(12)]
        
    return list(best_result)
