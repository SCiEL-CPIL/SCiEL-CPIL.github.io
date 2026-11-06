"""Strategy: Binned-mean denoising with Weighted Least Squares polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 10:
        return np.zeros(6)

    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.mean(y)])
        
    # Adaptive bin count
    # Aim for ~50 points per bin to balance bias/variance
    target_per_bin = 50
    n_bins = max(20, min(200, n // target_per_bin))
    
    # Sort x for binning
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Define bins
    edges = np.linspace(x_min, x_max, n_bins + 1)
    
    # Assign points to bins
    bin_indices = np.digitize(x_sorted, edges) - 1
    
    # Filter valid bins
    valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
    x_valid = x_sorted[valid_bins]
    y_valid = y_sorted[valid_bins]
    bin_indices_valid = bin_indices[valid_bins]
    
    # Compute bin statistics
    counts = np.bincount(bin_indices_valid, minlength=n_bins)
    sum_y = np.bincount(bin_indices_valid, weights=y_valid, minlength=n_bins)
    sum_x = np.bincount(bin_indices_valid, weights=x_valid, minlength=n_bins)
    
    non_empty = counts > 0
    bin_centers = np.zeros(n_bins)
    bin_means = np.zeros(n_bins)
    
    bin_centers[non_empty] = sum_x[non_empty] / counts[non_empty]
    bin_means[non_empty] = sum_y[non_empty] / counts[non_empty]
    
    fit_x = bin_centers[non_empty]
    fit_y = bin_means[non_empty]
    weights = counts[non_empty]
    
    if len(fit_x) < 6:
        # Fallback to simple linear fit on raw data if binning failed
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 5-len(coeff)))
        except Exception:
            return np.zeros(6)

    # Weighted Least Squares on binned means
    # Center and scale x to improve conditioning of Vandermonde matrix
    mu_x = np.mean(fit_x)
    sigma_x = np.std(fit_x)
    if sigma_x < 1e-10:
        sigma_x = 1.0
        
    fit_x_scaled = (fit_x - mu_x) / sigma_x
    
    # Construct Vandermonde matrix for scaled x
    V = np.vander(fit_x_scaled, 6, increasing=True)
    
    # Incorporate weights
    sqrt_w = np.sqrt(weights)
    V_w = V * sqrt_w[:, None]
    y_w = fit_y * sqrt_w
    
    try:
        # Solve weighted least squares
        coeff_scaled, residuals, rank, s = np.linalg.lstsq(V_w, y_w, rcond=None)
        
        if not np.all(np.isfinite(coeff_scaled)):
            raise ValueError("Non-finite coefficients in scaled fit")
            
        # Transform coefficients from scaled x back to original x
        # p(t) = sum_{i=0}^5 c_i t^i, where t = (x - mu) / sigma
        # We want q_j such that sum q_j x^j = p((x-mu)/sigma)
        coeff_orig = np.zeros(6)
        
        for i in range(6):
            ci = coeff_scaled[i] / (sigma_x ** i)
            # Expand (x - mu)^i = sum_{j=0}^i binom(i, j) x^j (-mu)^(i-j)
            for j in range(i + 1):
                # binom(i, j)
                comb_val = math.comb(i, j)
                term = ci * comb_val * ((-mu_x) ** (i - j))
                coeff_orig[j] += term
                
        if not np.all(np.isfinite(coeff_orig)):
            raise ValueError("Non-finite coefficients after transformation")
            
        return coeff_orig

    except Exception:
        # Fallback to unweighted polyfit on binned data
        try:
            coeff = np.polynomial.polynomial.polyfit(fit_x, fit_y, 5)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
            
        # Final fallback: linear fit on raw data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 5-len(coeff)))
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
