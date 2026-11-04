import numpy as np
import math

def construct():
    n = 12
    alpha = 2 * n - 1  # 23
    
    # Compute coefficients of L_n^(alpha)(x) = sum_{k=0}^n c_k x^k
    # c_k = (-1)^k * comb(n+alpha, n-k) / factorial(k)
    coeffs_asc = []
    for k in range(n + 1):
        # Calculate binomial coefficient
        # comb(n+alpha, n-k)
        binom_val = math.comb(n + alpha, n - k)
        fact_val = math.factorial(k)
        c_k = ((-1) ** k) * binom_val / fact_val
        coeffs_asc.append(c_k)
        
    # coeffs_asc is [c_0, c_1, ..., c_n]
    # np.polyroots expects [c_n, ..., c_1, c_0]
    coeffs_desc = coeffs_asc[::-1]
    
    # Find roots
    roots = np.polyroots(coeffs_desc)
    
    # Filter for real, positive roots
    # The roots of Laguerre polynomials are real and positive.
    # Due to numerical error, imaginary parts might be tiny.
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_val = r.real
            if real_val > 0:
                real_roots.append(real_val)
                
    # We expect exactly n positive roots
    if len(real_roots) != n:
        # Fallback to scipy if numpy fails or gives wrong count
        from scipy.special import roots_laguerre
        roots = roots_laguerre(n, alpha)
        real_roots = [float(r) for r in roots if r > 0]
        
    real_roots = np.sort(real_roots)
    return [float(r) for r in real_roots]
