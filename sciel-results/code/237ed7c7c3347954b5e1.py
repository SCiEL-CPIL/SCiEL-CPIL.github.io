"""Rewritten program for unresolved-tendency closure task using Chebyshev basis."""
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, with safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n < 6:
        return np.zeros(6, dtype=np.float64)
    
    # Normalize x to [-1, 1] for Chebyshev basis
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    x_mid = (x_max + x_min) / 2.0
    x_half_range = x_range / 2.0
    
    # Map x to t in [-1, 1]
    t = (x - x_mid) / x_half_range
    
    # Build Chebyshev basis matrix T(t) for degrees 0-5
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    T = np.zeros((n, 6), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, 6):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Check elapsed time
    if time.time() - start_time > DEADLINE - 5:
        # Fallback to linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.array([coeff[0], coeff[1], 0, 0, 0, 0], dtype=np.float64)
        except:
            return np.zeros(6, dtype=np.float64)
    
    # Ridge regression in Chebyshev basis
    # We want to find coefficients a such that y ≈ T @ a
    # Minimize ||T @ a - y||^2 + lambda * ||a||^2
    # (T^T T + lambda * I) a = T^T y
    
    TtT = T.T @ T
    Tty = T.T @ y
    I = np.eye(6)
    
    # Try a range of regularization parameters
    # Since we're in Chebyshev basis, the conditioning is much better
    # so we can use smaller lambda values
    lambdas = np.logspace(-8, 6, 25)
    
    best_a = np.zeros(6)
    best_score = np.inf
    
    # Precompute for speed
    for lam in lambdas:
        if time.time() - start_time > DEADLINE - 2:
            break
        try:
            A = TtT + lam * I
            a = np.linalg.solve(A, Tty)
            
            if not np.all(np.isfinite(a)):
                continue
            
            # Compute training MSE
            pred = T @ a
            mse = np.mean((pred - y) ** 2)
            
            # Use a simple model selection criterion
            # We want to balance fit and complexity
            # Since Chebyshev coefficients naturally decay for smooth functions,
            # we can penalize the magnitude of higher-order coefficients
            # AIC-like: n * log(MSE) + 2 * k where k is effective degrees of freedom
            # For ridge, effective df = sum of diagonal of (TtT + lam*I)^{-1} TtT
            try:
                A_inv = np.linalg.inv(A)
                df = np.trace(A_inv @ TtT)
            except:
                df = 6.0
            
            # Use a weighted score that penalizes high-order Chebyshev coefficients
            # This helps avoid overfitting to noise
            weights = np.array([1.0, 1.0, 2.0, 5.0, 10.0, 20.0])
            weighted_coeff_norm = np.sum(weights * a**2)
            
            score = mse + 0.01 * weighted_coeff_norm
            
            if score < best_score:
                best_score = score
                best_a = a
        except Exception:
            continue
    
    if not np.all(np.isfinite(best_a)):
        # Fallback to simple linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.array([coeff[0], coeff[1], 0, 0, 0, 0], dtype=np.float64)
        except:
            return np.zeros(6, dtype=np.float64)
    
    # Convert Chebyshev coefficients to monomial basis
    # We have y = sum_{k=0}^{5} a_k * T_k(t)
    # where t = (x - x_mid) / x_half_range
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    #
    # Substitute t = (x - x_mid) / x_half_range = alpha * x + beta
    # where alpha = 1/x_half_range, beta = -x_mid/x_half_range
    #
    # We need to express each T_k(t) as a polynomial in x
    # T_k(t) = sum_{j=0}^{k} c_{k,j} * t^j
    # Then substitute t = alpha*x + beta and expand
    
    # First, get the expansion of T_k(t) in powers of t
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    # Expansion coefficients for T_k(t) = sum_j C[k,j] * t^j
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    C[2, 0] = -1.0
    C[2, 2] = 2.0
    C[3, 1] = -3.0
    C[3, 3] = 4.0
    C[4, 0] = 1.0
    C[4, 2] = -8.0
    C[4, 4] = 8.0
    C[5, 1] = 5.0
    C[5, 3] = -20.0
    C[5, 5] = 16.0
    
    # Now t = alpha * x + beta
    # t^j = (alpha*x + beta)^j = sum_{m=0}^{j} binom(j, m) * alpha^m * x^m * beta^{j-m}
    
    alpha = 1.0 / x_half_range
    beta = -x_mid / x_half_range
    
    # c[j] = coefficient of x^j in the final polynomial
    c = np.zeros(6, dtype=np.float64)
    
    for k in range(6):
        a_k = best_a[k]
        if a_k == 0.0:
            continue
        
        for j in range(k + 1):
            C_kj = C[k, j]
            if C_kj == 0.0:
                continue
            
            # Expand (alpha*x + beta)^j
            for m in range(j + 1):
                # binom(j, m)
                if m == 0 or m == j:
                    binom_val = 1.0
                else:
                    # Compute binomial coefficient
                    binom_val = 1.0
                    for r in range(1, min(m, j - m) + 1):
                        binom_val = binom_val * (j - r + 1) / r
                
                term = C_kj * a_k * binom_val * (alpha ** m) * (beta ** (j - m))
                c[m] += term
    
    # Ensure finite
    if not np.all(np.isfinite(c)):
        return np.zeros(6, dtype=np.float64)
    
    return c.astype(np.float64)
