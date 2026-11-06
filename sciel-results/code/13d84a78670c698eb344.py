import numpy as np

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Filter out any NaNs or infinities just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        # Fallback to linear if data is too sparse
        coeff = np.zeros(6)
        if len(x) >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # Binning strategy for smoothing (approximating NW regression)
    # Choose number of bins. 
    # Too few bins -> high bias. Too many -> high variance (noise).
    # Rule of thumb: sqrt(N) or log(N). Let's try a few and pick best training MSE.
    
    n_samples = len(x)
    candidate_bins = [50, 100, 200, 500]
    # Adjust bins if N is small
    candidate_bins = [min(b, max(10, n_samples // 10)) for b in candidate_bins]
    candidate_bins = list(set(candidate_bins))
    
    best_coeff = None
    best_mse = np.inf
    
    for n_bins in candidate_bins:
        # Create bins
        # Use range of x
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-10:
            # All x are same, fit constant
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
            
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        
        # Digitize x into bins
        bin_indices = np.digitize(x, bin_edges) - 1
        # Handle edge cases where x == x_max might go out of bounds
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute mean x and mean y for each bin
        sum_x = np.bincount(bin_indices, weights=x, minlength=n_bins)
        count = np.bincount(bin_indices, minlength=n_bins)
        sum_y = np.bincount(bin_indices, weights=y, minlength=n_bins)
        
        # Avoid division by zero
        valid_bins = count > 0
        if np.sum(valid_bins) < 6: # Need at least 6 points for degree 5
            continue
            
        mean_x = np.zeros(n_bins)
        mean_y = np.zeros(n_bins)
        
        mean_x[valid_bins] = sum_x[valid_bins] / count[valid_bins]
        mean_y[valid_bins] = sum_y[valid_bins] / count[valid_bins]
        
        x_smooth = mean_x[valid_bins]
        y_smooth = mean_y[valid_bins]
        
        # Fit polynomial to smoothed data
        # Use degree 5
        try:
            coeff = np.polynomial.polynomial.polyfit(x_smooth, y_smooth, 5)
        except Exception:
            continue
            
        # Evaluate MSE on the original training data to select the best smoothing
        # This is a proxy for generalization error
        y_pred = np.polynomial.polynomial.polyval(x, coeff)
        mse = np.mean((y - y_pred) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_coeff = coeff
            
    if best_coeff is None:
        # Fallback to raw polyfit
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            best_coeff = np.zeros(6)
            if len(x) >= 2:
                best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            else:
                best_coeff[0] = np.mean(y) if len(y)>0 else 0.0

    return best_coeff
