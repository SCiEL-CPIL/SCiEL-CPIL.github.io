"""
Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
Uses normalized monomial basis with Ridge regression for stability.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Approach:
    1. Normalize x to zero mean, unit variance.
    2. Construct Vandermonde matrix for degree 5.
    3. Use Ridge Regression (L2 regularization) to find coefficients.
       Regularization strength is chosen to be small but non-zero to ensure 
       numerical stability without significantly biasing the fit.
    4. Transform coefficients from normalized space back to original x space.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 6:
        # Not enough points for degree-5. Try lower degrees or return zeros.
        # For n < 6, we can't fit degree 5 uniquely. 
        # The task implies we should return 6 coefficients. 
        # If we can't fit, returning zeros is a safe fallback, or a lower degree padded with zeros.
        if n >= 2:
            try:
                # Fit linear, pad to 6
                c_lin = np.polyfit(x, y, 1)
                coeff = np.zeros(6)
                coeff[0] = c_lin[1]
                coeff[1] = c_lin[0]
                return coeff
            except:
                pass
        return np.zeros(6)

    # 1. Normalize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # 2. Build Vandermonde matrix for normalized x (ascending powers)
    # V[i, j] = x_norm[i] ** j
    degree = 5
    V = np.ones((n, degree + 1))
    for i in range(1, degree + 1):
        V[:, i] = V[:, i-1] * x_norm

    # 3. Ridge Regression
    # We want to minimize ||V*c - y||^2 + lambda * ||c||^2
    # Solution: c = (V^T V + lambda * I)^-1 V^T y
    
    # Choose lambda. 
    # A common heuristic is lambda = alpha * trace(V^T V) / (degree+1)
    # Or simply a small fixed value. 
    # Since x is normalized, the diagonal of V^T V is roughly n.
    # Let's try a few small lambdas and pick the one with lowest residual on training set?
    # No, that's overfitting. 
    # Let's use a very small lambda to ensure stability, e.g., 1e-8.
    # Actually, for smooth physical functions, unregularized OLS on normalized data is often fine 
    # if n is large enough. But if n is small, regularization helps.
    
    # Let's try OLS first (lambda=0) and check condition number.
    # If ill-conditioned, use Ridge.
    
    try:
        # Calculate Gram matrix
        VtV = V.T @ V
        Vty = V.T @ y
        
        # Check condition number of VtV
        cond = np.linalg.cond(VtV)
        
        if cond < 1e10:
            # OLS is stable
            coeffs_norm = np.linalg.solve(VtV, Vty)
        else:
            # Use Ridge with small lambda
            # lambda proportional to the scale of VtV
            lam = 1e-6 * np.trace(VtV) / (degree + 1)
            A = VtV + lam * np.eye(degree + 1)
            coeffs_norm = np.linalg.solve(A, Vty)
            
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        coeffs_norm = np.linalg.lstsq(V, y, rcond=None)[0]

    if not np.all(np.isfinite(coeffs_norm)):
        return np.zeros(6)

    # 4. Transform coefficients from normalized space to original space
    # We have y = sum_{i=0}^5 c_i_norm * (x_norm)^i
    # x_norm = (x - x_mean) / x_std
    # So y = sum_{i=0}^5 c_i_norm * (1/x_std)^i * (x - x_mean)^i
    # Let a_i = c_i_norm / x_std^i
    # y = sum_{i=0}^5 a_i * (x - x_mean)^i
    # Expand (x - x_mean)^i = sum_{k=0}^i binom(i, k) x^k (-x_mean)^(i-k)
    # y = sum_{k=0}^5 [ sum_{i=k}^5 a_i * binom(i, k) * (-x_mean)^(i-k) ] x^k
    
    coeff_orig = np.zeros(6)
    from math import comb
    
    for i in range(degree + 1):
        a_i = coeffs_norm[i] / (x_std ** i)
        for k in range(i + 1):
            coeff_orig[k] += a_i * comb(i, k) * ((-x_mean) ** (i - k))
            
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback
        return np.zeros(6)
        
    return coeff_orig

# EVOLVE-BLOCK-END
