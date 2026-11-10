"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fit_centered(x, y, degree, lam):
    """
    Fit a polynomial of given degree to centered/normalized data using Ridge Regression.
    Returns coefficients in the original basis.
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)
    
    x_norm = (x - x_mean) / x_std
    
    # Construct Vandermonde matrix for centered/normalized x
    V = np.vander(x_norm, N=degree+1, increasing=True)
    
    # Ridge regression: (V^T V + lam * I) c = V^T y
    # We add small regularization to diagonal except for intercept (index 0)
    I = np.eye(degree+1)
    I[0,0] = 0  # No regularization on intercept
    
    try:
        A = V.T @ V + lam * I
        b = V.T @ y
        c_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        try:
            c_norm, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        except:
            return np.zeros(6)
    
    # Convert coefficients from normalized basis to original basis
    # y = sum_{k=0}^{degree} c_k * ((x - x_mean)/x_std)^k
    # We need to expand this into sum_{i=0}^{5} a_i * x^i
    # This is a bit involved. Let's use a simpler approach:
    # Just fit directly in original basis but with centered x?
    # No, the task asks for coefficients in ascending powers of local slow state x.
    # So we must return a_i such that sum a_i x^i approximates y.
    
    # Let's re-do: Fit in original basis but use the same regularization structure?
    # Or better: Convert c_norm to a_i.
    # Let's use polynomial composition.
    # p(x_norm) = sum c_k x_norm^k
    # x_norm = (x - m)/s
    # p(x) = sum c_k ((x-m)/s)^k = sum c_k s^-k (x-m)^k
    # (x-m)^k = sum_{j=0}^k binom(k,j) x^j (-m)^(k-j)
    
    a = np.zeros(6)
    for k in range(degree+1):
        coeff_k = c_norm[k] / (x_std ** k)
        # Expand (x - x_mean)^k
        # Using binomial theorem
        for j in range(k+1):
            binom_coeff = 1.0
            # Compute binom(k, j)
            if k == 0:
                binom_coeff = 1.0
            else:
                # Simple calculation for small k
                num = 1
                for i in range(j):
                    num *= (k - i)
                den = 1
                for i in range(1, j+1):
                    den *= i
                binom_coeff = num / den
            
            a[j] += coeff_k * binom_coeff * ((-x_mean) ** (k-j))
    
    # Ensure we only return up to degree 5
    if degree < 5:
        # Zero out higher coefficients
        for i in range(degree+1, 6):
            a[i] = 0.0
    
    return a

def _loo_cv_mse(x, y, degree, lam):
    """
    Compute Leave-One-Out CV MSE for a given degree and lambda.
    """
    n = len(x)
    if n < 10:
        return np.inf
    
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return np.inf
    
    x_norm = (x - x_mean) / x_std
    V = np.vander(x_norm, N=degree+1, increasing=True)
    
    I = np.eye(degree+1)
    I[0,0] = 0
    
    try:
        A = V.T @ V + lam * I
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    
    # Hat matrix H = V (V^T V + lam I)^-1 V^T
    # H_ii = V_i A_inv V_i^T
    # MSE_LOO = sum (y_i - y_hat_i_loo)^2 / n
    # y_hat_i_loo = (y - (1 - h_ii) y_i) / (1 - h_ii) ? No.
    # y_hat_loo_i = (y - (1/h_ii) (y - y_hat_full_i)) ... 
    # Standard formula: y_loo_i = (y_full_i - h_ii y_i) / (1 - h_ii)
    
    try:
        H_diag = np.einsum('ij,ji->i', V, A_inv @ V.T)
        # Numerical stability
        H_diag = np.clip(H_diag, 0, 1 - 1e-6)
        
        y_full = V @ (A_inv @ (V.T @ y))
        
        y_loo = (y_full - H_diag * y) / (1 - H_diag)
        
        mse = np.mean((y - y_loo) ** 2)
        return mse
    except:
        return np.inf

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    if np.std(x) < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)
    
    # Search over degrees and regularization strengths
    degrees = [3, 4, 5]
    # Log-spaced lambda values
    log_lams = np.linspace(np.log(1e-4), np.log(1e2), 20)
    lams = np.exp(log_lams)
    
    best_mse = np.inf
    best_params = (5, 1e-2)
    
    # Evaluate each combination
    for d in degrees:
        if time.time() > deadline - 10:
            break
        for lam in lams:
            if time.time() > deadline - 10:
                break
            mse = _loo_cv_mse(x, y, d, lam)
            if mse < best_mse:
                best_mse = mse
                best_params = (d, lam)
    
    d, lam = best_params
    
    # Final fit with best parameters
    coeff = _fit_centered(x, y, d, lam)
    
    if not np.all(np.isfinite(coeff)):
        # Fallback
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except:
            coeff = np.zeros(6)
    
    return coeff

# EVOLVE-BLOCK-END
