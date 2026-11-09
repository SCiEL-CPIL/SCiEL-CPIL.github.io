import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    if n < 6:
        deg = min(5, n - 1)
        coeff = np.zeros(6)
        if deg >= 0:
            coeff[:deg+1] = np.polynomial.polynomial.polyfit(x, y, deg)
        return coeff
    
    # Strategy: fit using Chebyshev basis for numerical stability,
    # then convert to power basis.
    # Map x to [-1, 1] using min/max for Chebyshev fit
    
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    
    # Map to [-1, 1]
    x_cheb = (2.0 * x - (x_max + x_min)) / x_range
    
    # Fit degree-5 polynomial in Chebyshev-like basis (power basis on mapped x)
    # y = sum_{j=0}^{5} c_j * x_cheb^j
    try:
        coeff_cheb = np.polynomial.polynomial.polyfit(x_cheb, y, 5)
    except Exception:
        coeff_cheb = np.zeros(6)
    
    # Convert from power basis on x_cheb to power basis on x
    # x_cheb = (2*x - (x_max+x_min)) / x_range = (2/x_range)*x - (x_max+x_min)/x_range
    # So x_cheb = a*x + b where a = 2/x_range, b = -(x_max+x_min)/x_range
    a = 2.0 / x_range
    b = -(x_max + x_min) / x_range
    
    # (a*x + b)^j = sum_{k=0}^{j} C(j,k) * a^k * b^(j-k) * x^k
    coeff_orig = np.zeros(6)
    for j in range(6):
        c_j = coeff_cheb[j]
        for k in range(j + 1):
            # C(j, k)
            if k == 0 or k == j:
                binom = 1.0
            else:
                binom = 1.0
                for m in range(k):
                    binom *= (j - m) / (m + 1)
            coeff_orig[k] += c_j * binom * (a ** k) * (b ** (j - k))
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return coeff_orig
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback: direct polyfit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return np.zeros(6)
    
    return coeff_orig
