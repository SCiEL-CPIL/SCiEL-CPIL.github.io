import numpy as np

def construct():
    n = 12
    alpha = 1.0
    
    # Coefficients of L_n^(alpha)(x)
    # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
    # We need coefficients for x^0 to x^n
    coeffs = np.zeros(n+1)
    for k in range(n+1):
        # binom(n+alpha, n-k)
        # Using gamma function for non-integer alpha, but alpha=1 is integer
        # binom(12+1, 12-k) = binom(13, 12-k)
        # k! = factorial(k)
        # We can use scipy.special.comb or math.comb
        from math import comb, factorial
        binom_val = comb(n + int(alpha), n - k) if alpha == int(alpha) else 1.0 # Fallback for non-integer
        # For alpha=1: binom(13, 12-k)
        # Let's use scipy.special.comb for safety with alpha
        from scipy.special import comb as scomb
        binom_val = scomb(n + alpha, n - k, exact=False)
        fact_val = factorial(k)
        coeffs[k] = (-1)**k * binom_val / fact_val
        
    # Find roots
    roots = np.roots(coeffs)
    # Filter real positive roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    
    return [float(r) for r in real_roots]
