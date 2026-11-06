"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge cases: if data is empty or constant
    if len(x) == 0 or np.std(x) < 1e-12:
        # If x is constant, we can only fit a constant term (c0)
        if len(y) > 0:
            return np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)
        return np.zeros(6, dtype=np.float64)
    
    # 1. Normalize x to improve numerical stability for higher-degree polynomial fitting
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    x_norm = (x - mu) / sigma
    
    # 2. Fit a degree-5 polynomial using the normalized variable
    # We use least squares via Vandermonde matrix or polyfit
    # polyfit is generally stable enough for degree 5 if input is normalized
    try:
        # Coefficients for Q(x_norm) = sum_{i=0}^5 q_i (x_norm)^i
        q_coeffs = np.polynomial.polynomial.polyfit(x_norm, y, 5)
    except Exception:
        # Fallback to linear if higher degree fails (should not happen usually)
        q_coeffs = np.polynomial.polynomial.polyfit(x_norm, y, 1)
        # Pad with zeros
        if len(q_coeffs) < 6:
            q_coeffs = np.pad(q_coeffs, (0, 6 - len(q_coeffs)))
    
    # 3. Convert coefficients from Q(x_norm) to P(x)
    # x_norm = (x - mu) / sigma
    # P(x) = Q((x - mu) / sigma)
    # Let z = x - mu. Then Q(z/sigma) = sum q_i (z/sigma)^i = sum (q_i / sigma^i) z^i
    # z^i = (x - mu)^i = sum_{j=0}^i binomial(i, j) x^j (-mu)^(i-j)
    
    # We want P(x) = sum_{j=0}^5 c_j x^j
    # c_j = sum_{i=j}^5 (q_i / sigma^i) * binomial(i, j) * (-mu)^(i-j)
    
    c_coeffs = np.zeros(6, dtype=np.float64)
    for i in range(6):
        if i >= len(q_coeffs):
            q_i = 0.0
        else:
            q_i = q_coeffs[i]
        
        if q_i == 0.0:
            continue
            
        factor = q_i / (sigma ** i)
        
        for j in range(i + 1):
            # Binomial coefficient C(i, j)
            binom = 1.0
            # Calculate C(i, j)
            # C(i, j) = i! / (j! (i-j)!)
            # For small i, we can compute directly
            n = i
            k = j
            # Optimization: C(i, j) = C(i, i-j)
            if k > n - k:
                k = n - k
            if k == 0:
                binom = 1.0
            else:
                binom = 1.0
                for m in range(k):
                    binom *= (n - m)
                    binom /= (m + 1)
            
            term = factor * binom * ((-mu) ** (i - j))
            c_coeffs[j] += term
            
    # Check time
    if time.time() > deadline:
        # If we are running out of time, return what we have (it should be ready)
        pass
        
    return c_coeffs
# EVOLVE-BLOCK-END
