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
        # Not enough data for degree 5, use lower degree
        deg = min(5, n - 1)
        coeff = np.zeros(6)
        if deg >= 0:
            coeff[:deg+1] = np.polynomial.polynomial.polyfit(x, y, deg)
        return coeff
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for degree 5 in scaled x
    # V[i, j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Solve least squares
    try:
        coeff_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to polyfit
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        return coeff
    
    # Convert from scaled polynomial to original polynomial
    # If y = sum(a_j * ((x - x_mean)/x_std)^j)
    # Then expand: ((x - x_mean)/x_std)^j = (1/x_std^j) * (x - x_mean)^j
    # And (x - x_mean)^j = sum_{k=0}^{j} C(j,k) * x^k * (-x_mean)^(j-k)
    
    coeff_orig = np.zeros(6)
    
    for j in range(6):
        a_j = coeff_scaled[j] / (x_std ** j)
        # Expand (x - x_mean)^j
        for k in range(j + 1):
            binom = 1
            # Compute C(j, k)
            if k < j:
                # C(j, k) = j! / (k! * (j-k)!)
                num = 1
                den = 1
                for m in range(k):
                    num *= (j - m)
                for m in range(1, k+1):
                    den *= m
                binom = num / den
            else:
                binom = 1.0
            coeff_orig[k] += a_j * binom * ((-x_mean) ** (j - k))
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return what we have (shouldn't happen for simple lstsq)
        return coeff_orig
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        return coeff
    
    return coeff_orig
