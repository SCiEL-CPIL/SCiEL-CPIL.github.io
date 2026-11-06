import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty for this, 150s is the hard limit
    
    # Ensure inputs are 1D numpy arrays
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Filter out any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Strategy: Two-stage approach
    # Stage 1: Denoise by binning and averaging
    # Stage 2: Fit degree-5 polynomial to the binned means
    
    # Determine bin count. 
    # Too few bins: high bias. Too many: high variance (similar to raw).
    # 100 bins is a reasonable compromise for "smoothing".
    n_bins = 100
    try:
        # np.histogram returns bin edges. We need centers.
        counts, bin_edges = np.histogram(x, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Calculate mean y for each bin
        # Only use bins with at least one sample
        valid_bins = counts > 0
        if not np.any(valid_bins):
            # Fallback to simple OLS if binning fails
            return np.polynomial.polynomial.polyfit(x, y, 5)
            
        x_binned = bin_centers[valid_bins]
        weights = counts[valid_bins]
        
        # Compute weighted mean of y for each bin
        # We need to sum y for each bin and divide by count
        # np.digitize or np.searchsorted to assign y to bins
        # Using np.histogram with weights is efficient
        sums, _ = np.histogram(x, bins=n_bins, weights=y)
        y_binned = sums[valid_bins] / weights
        
    except Exception:
        # Fallback to simple OLS if binning fails
        return np.polynomial.polynomial.polyfit(x, y, 5)
    
    # Check time
    if time.time() > deadline:
        # If we ran out of time (unlikely), return best guess so far or zeros
        # But we are in the middle of calculation. Let's just proceed, it's fast.
        pass

    # Stage 2: Fit polynomial to binned data
    # We use weighted least squares.
    # np.polynomial.polynomial.polyfit does not support weights directly in older numpy?
    # Actually, np.polyfit supports weights. np.polynomial.polynomial.polyfit does NOT support weights in standard interface?
    # Let's check: np.polynomial.polynomial.polyfit(x, y, deg, full=False, w=None) -> w IS supported in recent numpy.
    
    try:
        # Fit degree 5
        coeff = np.polynomial.polynomial.polyfit(x_binned, y_binned, 5, w=weights)
        
        # Check for finite coefficients
        if not np.all(np.isfinite(coeff)):
             # Fallback to unweighted OLS on raw data
             coeff = np.polynomial.polynomial.polyfit(x, y, 5)
             
        return coeff
        
    except Exception:
        # Fallback to linear fit if degree 5 fails
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            out = np.zeros(6)
            out[:2] = coeff
            return out
        except Exception:
            return np.zeros(6)
