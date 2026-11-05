"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        # Not enough data for degree-5 fit
        return np.zeros(6, dtype=np.float64)
    
    # Build Vandermonde matrix for degree-5 polynomial: [1, x, x^2, x^3, x^4, x^5]
    powers = np.arange(6)
    V = x[:, None] ** powers  # shape (n, 6)
    
    # Check elapsed time
    if time.time() - start_time > DEADLINE - 5:
        # Fallback to linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.array([coeff[0], coeff[1], 0, 0, 0, 0], dtype=np.float64)
        except:
            return np.zeros(6, dtype=np.float64)
    
    # Try degree-5 least squares fit with regularization
    # Use SVD for numerical stability
    try:
        # Add small regularization to handle potential ill-conditioning
        reg = 1e-8
        VtV = V.T @ V + reg * np.eye(6)
        Vty = V.T @ y
        coeff = np.linalg.solve(VtV, Vty)
        
        # Verify coefficients are finite
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
            
    except Exception:
        # Fallback: try lower degree fits
        try:
            coeff_full = np.zeros(6)
            for deg in [4, 3, 2, 1, 0]:
                try:
                    c = np.polynomial.polynomial.polyfit(x, y, deg)
                    coeff_full[:deg+1] = c
                    if np.all(np.isfinite(coeff_full)):
                        break
                except:
                    continue
            return coeff_full
        except:
            return np.zeros(6, dtype=np.float64)
    
    # Check elapsed time
    if time.time() - start_time > DEADLINE - 3:
        return coeff
    
    # Try to improve with weighted least squares, emphasizing regions with more data
    # and trying different regularization strengths
    best_coeff = coeff
    best_mse = np.mean((V @ coeff - y) ** 2)
    
    # Try a few regularization strengths
    for reg_strength in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        if time.time() - start_time > DEADLINE - 2:
            break
        try:
            VtV_reg = V.T @ V + reg_strength * np.eye(6)
            Vty_reg = V.T @ y
            coeff_reg = np.linalg.solve(VtV_reg, Vty_reg)
            if np.all(np.isfinite(coeff_reg)):
                mse_reg = np.mean((V @ coeff_reg - y) ** 2)
                if mse_reg < best_mse:
                    best_mse = mse_reg
                    best_coeff = coeff_reg
        except:
            continue
    
    # Also try truncated SVD approach for potentially better conditioning
    try:
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
        # Use all singular values but with Tikhonov regularization
        for lam in [1e-7, 1e-6, 1e-5, 1e-4]:
            if time.time() - start_time > DEADLINE - 1:
                break
            try:
                s_reg = s / (s**2 + lam**2)
                coeff_svd = Vt.T @ (s_reg * (U.T @ y))
                if np.all(np.isfinite(coeff_svd)):
                    mse_svd = np.mean((V @ coeff_svd - y) ** 2)
                    if mse_svd < best_mse:
                        best_mse = mse_svd
                        best_coeff = coeff_svd
            except:
                continue
    except:
        pass
    
    # Final check
    if not np.all(np.isfinite(best_coeff)):
        return np.zeros(6, dtype=np.float64)
    
    return best_coeff.astype(np.float64)
# EVOLVE-BLOCK-END
