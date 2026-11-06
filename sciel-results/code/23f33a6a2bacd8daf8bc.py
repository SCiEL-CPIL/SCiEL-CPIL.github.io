# EVOLVE-BLOCK-START
import numpy as np
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation using Robust Regression
    with Huber Loss via Iteratively Reweighted Least Squares (IRLS).
    
    Args:
        training_x: array-like, flattened slow-state values
        training_subgrid_tendency: array-like, flattened subgrid tendency values
    
    Returns:
        np.ndarray of shape (6,) with coefficients in ASCENDING powers of x
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Compute statistics for scaling
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    
    # Scale x and y for numerical stability
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    
    # Build Vandermonde matrix for degree 5
    def make_vandermonde(x_vals, deg=5):
        powers = np.arange(deg + 1)
        return x_vals[:, None] ** powers
    
    V = make_vandermonde(x_scaled, 5)
    n = len(x_scaled)
    
    # Initialize with OLS
    try:
        coeff_scaled, _, _, _ = np.linalg.lstsq(V, y_scaled, rcond=None)
    except:
        coeff_scaled = np.zeros(6)
    
    # IRLS Parameters for Huber Loss
    # delta is the threshold where loss changes from quadratic to linear
    # Typical values are 1.345 for 95% efficiency
    delta = 1.345
    max_iter = 100
    tol = 1e-6
    
    for _ in range(max_iter):
        # Compute residuals
        y_pred = V @ coeff_scaled
        residuals = y_scaled - y_pred
        
        # Estimate scale (MAD)
        med_r = np.median(residuals)
        mad_r = np.median(np.abs(residuals - med_r))
        if mad_r < 1e-12:
            scale = 1.0
        else:
            scale = 1.4826 * mad_r
        
        # Compute Huber weights
        # u = residual / scale
        u = residuals / scale
        abs_u = np.abs(u)
        
        # Huber weight: 1 if |u| <= delta, else delta / |u|
        weights = np.ones(n)
        outlier_mask = abs_u > delta
        if np.any(outlier_mask):
            weights[outlier_mask] = delta / abs_u[outlier_mask]
            
        # Weighted Least Squares
        W = np.diag(weights)
        
        try:
            A = V.T @ W @ V
            b = V.T @ W @ y_scaled
            
            # Add small ridge for numerical stability
            A += 1e-10 * np.eye(6)
            
            new_coeff_scaled = np.linalg.solve(A, b)
            
            # Check convergence
            if np.linalg.norm(new_coeff_scaled - coeff_scaled) < tol:
                coeff_scaled = new_coeff_scaled
                break
                
            coeff_scaled = new_coeff_scaled
            
        except np.linalg.LinAlgError:
            # Fallback to OLS if weighted solve fails
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V, y_scaled, rcond=None)
            except:
                coeff_scaled = np.zeros(6)
            break

    # coeff_scaled is in terms of x_scaled and y_scaled
    # y = y_std * P(x_scaled) + y_mean
    # P(x_scaled) = sum c_i ((x - x_mean) / x_std)^i
    # y = y_std * sum c_i / x_std^i * (x - x_mean)^i + y_mean
    
    # Expand (x - x_mean)^i into powers of x
    coeff_original = np.zeros(6)
    
    # The constant term from y_mean
    coeff_original[0] += y_mean
    
    for i in range(6):
        c_i = coeff_scaled[i]
        if abs(c_i) < 1e-15:
            continue
        
        # Term is y_std * c_i / x_std^i * (x - x_mean)^i
        scale_factor = y_std * c_i / (x_std ** i)
        
        # Expand (x - x_mean)^i = sum_{j=0}^i binom(i, j) x^j (-x_mean)^(i-j)
        for j in range(i + 1):
            coeff_original[j] += scale_factor * comb(i, j) * ((-x_mean) ** (i - j))
    
    # Clip to finite values
    coeff_original = np.where(np.isfinite(coeff_original), coeff_original, 0.0)
    
    return coeff_original
# EVOLVE-BLOCK-END
