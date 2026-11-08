"""Fourier-invariant Hermite-Gaussian test function tuned via analytic spectral moments."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_bound(a0, a4, a8):
    """
    Computes the upper bound on A(f)*A(F) for the given coefficients.
    a12 is derived to satisfy g(0)=0.
    """
    # Hermite values at 0
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    
    # Constraint: a0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    # a12 = -(a0 - 12*a4 + 1680*a8) / (-47880)
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 47880.0
    
    coeffs = np.array([a0, a4, a8, a12])
    k = np.array([0, 4, 8, 12], dtype=float)
    
    # Precompute weights w_k = 2^k k!
    # Use log to avoid overflow? No, 12! is small.
    fact = np.array([1.0, 24.0, 40320.0, 479001600.0]) # 0!, 4!, 8!, 12!
    w = (2.0 ** k) * fact
    
    # Denominator: Norm^2 = sum a_k^2 w_k
    norm_sq = np.sum(coeffs**2 * w)
    
    if norm_sq < 1e-12:
        return 1e100 # Avoid division by zero
        
    # Numerator for <x^2>: sum a_k^2 * (2k+1) * 2^k k! / 2 ?
    # <x^2> = sum a_k^2 * (k + 0.5) * w_k / norm_sq
    # Note: integral of x^2 H_k^2 e^{-x^2} is (k+0.5) * 2^k k! * sqrt(pi)
    # The sqrt(pi) cancels in the ratio.
    x2_num = np.sum(coeffs**2 * (k + 0.5) * w)
    x2 = x2_num / norm_sq
    
    # Numerator for <xi^2>: sum |g'|^2
    # g' = (P' - xP) e^{-x^2/2}
    # Q = P' - xP = sum a_k (k H_{k-1} - 0.5 H_{k+1})
    # We need sum b_j^2 w_j where b_j are coeffs of H_j in Q.
    # b_1 = -0.5 a_0
    # b_3 = 4 a_4
    # b_5 = -0.5 a_4
    # b_7 = 8 a_8
    # b_9 = -0.5 a_8
    # b_11 = 12 a_12
    # b_13 = -0.5 a_12
    
    # Weights for odd Hermite polynomials
    # w_1 = 2^1 1! = 2
    # w_3 = 2^3 3! = 48
    # w_5 = 2^5 5! = 3840
    # w_7 = 2^7 7! = 107520
    # w_9 = 2^9 9! = 5184000
    # w_11 = 2^11 11! = 2048 * 39916800 = 81759539200
    # w_13 = 2^13 13! = 8192 * 6227020800 = 51011754380800
    
    w_odd = {
        1: 2.0,
        3: 48.0,
        5: 3840.0,
        7: 107520.0,
        9: 5184000.0,
        11: 81759539200.0,
        13: 51011754380800.0
    }
    
    b1 = -0.5 * a0
    b3 = 4.0 * a4
    b5 = -0.5 * a4
    b7 = 8.0 * a8
    b9 = -0.5 * a8
    b11 = 12.0 * a12
    b13 = -0.5 * a12
    
    xi2_num = (b1**2 * w_odd[1] + 
               b3**2 * w_odd[3] + 
               b5**2 * w_odd[5] + 
               b7**2 * w_odd[7] + 
               b9**2 * w_odd[9] + 
               b11**2 * w_odd[11] + 
               b13**2 * w_odd[13])
    
    xi2 = xi2_num / norm_sq
    
    # The bound is A(f) A(F) = sqrt(x2) * sqrt(xi2) = sqrt(x2 * xi2)
    # We minimize the square of the bound to avoid sqrt in optimization
    return x2 * xi2

def construct():
    start = time.time()
    deadline = start + 170.0 # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.0003086])
    
    best_score = _compute_bound(*x0)
    best_x = x0
    
    # Use Nelder-Mead for local optimization
    try:
        res = minimize(_compute_bound, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # If time remains, try a few random restarts
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            # Random perturbation around best or random
            # Scale is important. a12 is small.
            x_try = best_x + np.random.normal(0, 0.1, size=3)
            # Ensure a12 doesn't blow up? It's linear, so fine.
            try:
                val = _compute_bound(*x_try)
                if val < best_score:
                    best_score = val
                    best_x = x_try
                    # Local refine
                    res = minimize(_compute_bound, x_try, method='Nelder-Mead', 
                                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500})
                    if res.success and res.fun < best_score:
                        best_score = res.fun
                        best_x = res.x
            except Exception:
                continue
    
    return best_x

# EVOLVE-BLOCK-END
