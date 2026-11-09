"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Strategy:
    # 1. Estimate E[Y|X] by aggregating data into bins of varying widths.
    # 2. For each bin width, compute the mean y in each bin.
    # 3. Fit a degree-5 polynomial to these (bin_center, bin_mean) pairs.
    # 4. Select the bin width that minimizes a score combining the fit residual
    #    on the aggregated data and a penalty for the number of effective bins
    #    (to avoid overfitting to noise in very fine bins).
    # 5. Return the coefficients of the best polynomial.
    
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.array([0, 0, 0, 0, 0, np.mean(y)], dtype=np.float64)
    
    # Define a range of bin counts to test.
    # We want to go from very coarse (low bias, high variance) to very fine (high bias, low variance).
    # Too few bins -> underfit. Too many bins -> overfit to noise.
    # Let's test bin counts from 10 to 200.
    
    best_coeff = None
    best_score = np.inf
    
    # Pre-compute Vandermonde matrices for the polynomial fit if possible?
    # No, the x values (bin centers) change.
    
    # To speed up, we can vectorize the binning process if we sort x once.
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Candidate number of bins
    # Logarithmic spacing of bin counts to cover a wide range efficiently
    bin_counts = np.unique(np.logspace(np.log10(10), np.log10(300), num=20).astype(int))
    
    for num_bins in bin_counts:
        if time.time() - t_start > deadline - 5.0:
            break
            
        # Bin the data
        # Use np.digitize or similar to assign bins
        bin_edges = np.linspace(x_min, x_max, num_bins + 1)
        
        # Assign each point to a bin
        # np.digitize returns indices 1..num_bins for values in [edge_i, edge_{i+1})
        # We want 0..num_bins-1
        bin_indices = np.digitize(x_sorted, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute bin means
        # Initialize sums and counts
        bin_sums = np.zeros(num_bins)
        bin_counts_arr = np.zeros(num_bins)
        
        # Use np.bincount for efficiency
        bin_sums = np.bincount(bin_indices, weights=y_sorted, minlength=num_bins)
        bin_counts_arr = np.bincount(bin_indices, minlength=num_bins)
        
        # Find valid bins (count > 0)
        valid_bins = bin_counts_arr > 0
        if np.sum(valid_bins) < 6:
            # Not enough data points for a degree 5 polynomial
            continue
            
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_means = np.zeros(num_bins)
        bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts_arr[valid_bins]
        
        x_fit = bin_centers[valid_bins]
        y_fit = bin_means[valid_bins]
        
        # Fit degree 5 polynomial
        # A = [1, x, x^2, x^3, x^4, x^5]
        A = np.vander(x_fit, 6, increasing=True)
        
        try:
            coeff, residuals, rank, sv = np.linalg.lstsq(A, y_fit, rcond=None)
        except Exception:
            continue
            
        if not np.all(np.isfinite(coeff)):
            continue
            
        # Calculate score
        # We want to minimize the error on the aggregated data, but penalize
        # overfitting.
        # Residual on aggregated data:
        y_pred = np.dot(A, coeff)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        
        # Penalty:
        # A simple heuristic: penalize based on the variance of the residuals
        # relative to the variance of the data?
        # Or just use a small penalty on the number of bins to prefer smoother fits
        # if the residual is similar.
        # Actually, since we are fitting to the *means*, the residual is already
        # "denoised". The main risk is that with too many bins, the means
        # themselves are noisy, and the polynomial overfits to those noisy means.
        # With too few bins, the means are biased.
        
        # Let's use a score that combines the residual sum of squares on the
        # aggregated data with a penalty term that increases with the number of bins.
        # The penalty should be small enough to allow refinement but large enough
        # to prevent overfitting.
        
        # A common approach is to use the AIC or BIC, but we don't have the
        # exact likelihood. A simple penalty is:
        # score = ss_res + lambda * num_bins
        # What is a good lambda?
        # The variance of the mean in a bin is roughly Var(Y) / count.
        # So the expected residual due to noise in the means is roughly
        # sum(Var(Y) / count_i) over bins.
        # Let's estimate Var(Y) from the data.
        
        # Instead of a complex penalty, let's look at the residual on the
        # *original* data, but weighted by the inverse of the bin variance?
        # No, that's just standard regression.
        
        # Let's try a different score: the mean squared error on the
        # aggregated data, normalized by the variance of the aggregated data.
        # And then add a penalty for the number of bins.
        
        var_y_fit = np.var(y_fit)
        if var_y_fit == 0:
            mse = ss_res
        else:
            mse = ss_res / (len(y_fit) * var_y_fit)
            
        # Penalty:
        # The number of effective parameters is 6 (coefficients).
        # The number of data points is len(y_fit).
        # A penalty of 0.1 * num_bins might be too strong or weak.
        # Let's try a penalty that is proportional to the number of bins
        # but scaled by the typical residual.
        
        # Actually, a simpler and robust method is to check the "elbow" in the
        # residual sum of squares as a function of num_bins.
        # But we want to do this in one pass.
        
        # Let's use a score: ss_res / len(y_fit) + 0.01 * num_bins * (ss_res / len(y_fit))
        # This is like a relative penalty.
        
        score = mse + 0.05 * (num_bins / 100.0)
        
        if score < best_score:
            best_score = score
            best_coeff = coeff
    
    if best_coeff is None:
        # Fallback to a simple global mean if no bin worked
        return np.array([0, 0, 0, 0, 0, np.mean(y)], dtype=np.float64)
        
    return best_coeff
# EVOLVE-BLOCK-END
