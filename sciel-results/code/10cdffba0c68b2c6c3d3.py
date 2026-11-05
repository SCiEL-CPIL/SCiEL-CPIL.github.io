# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial using a binned-mean strategy to denoise the training data.
    
    Strategy:
    1. Sort training data by x.
    2. Divide x into quantile-based bins to ensure robust handling of non-uniform x distribution.
    3. Compute the mean subgrid tendency in each bin. This creates a piecewise-constant approximation 
       of the underlying smooth function, effectively averaging out high-frequency noise.
    4. Fit a degree-5 polynomial to these binned means using weighted least squares (weights = bin counts).
    5. Return the polynomial coefficients in ascending powers of x.
    
    This approach mitigates the risk of the global polynomial overfitting local noise fluctuations 
    in the raw training pairs, leading to a smoother and more generalizable closure.
    """
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    
    if n < 6:
        # Not enough points for degree 5, fallback to lower degree
        deg = min(5, n - 1)
        if deg < 0:
            return np.zeros(6)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6)
            full_coeff[:deg+1] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6)
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        return np.zeros(6)

    # 1. Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # 2. Define bins
    # We want to smooth the data. Too many bins = no smoothing (similar to raw fit).
    # Too few bins = over-smoothing (loses detail).
    # Let's try ~20-50 bins.
    num_bins = 30
    
    # Use quantile bins to handle non-uniform x density
    # np.quantile with linspace
    edges = np.quantile(x_sorted, np.linspace(0, 1, num_bins + 1))
    
    # Ensure unique edges (handle duplicates in x)
    # If there are many duplicates, quantile might produce repeated edges.
    # We'll use np.digitize to assign points to bins.
    # np.digitize returns indices 0..num_bins. Points > max go to num_bins.
    # We want points in [edges[i], edges[i+1]) to be in bin i.
    # np.digitize(x, edges) returns i such that edges[i-1] <= x < edges[i].
    # So bin index = digitize - 1.
    
    bin_indices = np.digitize(x_sorted, edges) - 1
    # Clip to valid range [0, num_bins-1]
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # 3. Compute bin means and counts
    bin_sums = np.bincount(bin_indices, weights=y_sorted, minlength=num_bins)
    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    
    # Avoid division by zero
    valid_bins = bin_counts > 0
    if np.sum(valid_bins) < 6:
        # Fallback to raw fit if binning failed
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except Exception:
            return np.zeros(6)
            
    x_bin_centers = np.zeros(num_bins)
    y_bin_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        if valid_bins[i]:
            # Center of the bin in x-space
            # Use the mean of x in the bin for better representation
            mask = bin_indices == i
            x_bin_centers[i] = np.mean(x_sorted[mask])
            y_bin_means[i] = bin_sums[i] / bin_counts[i]
        else:
            # If a bin is empty, we can skip it or interpolate. 
            # For simplicity, we will just exclude it from the fit.
            pass
            
    # Extract valid data for the fit
    x_fit = x_bin_centers[valid_bins]
    y_fit = y_bin_means[valid_bins]
    w_fit = bin_counts[valid_bins]
    
    # 4. Weighted Least Squares Fit
    # We want to fit P(x) = sum c_i x^i to (x_fit, y_fit) with weights w_fit.
    # This is equivalent to minimizing sum w_i * (P(x_i) - y_i)^2
    
    try:
        # Construct the Vandermonde matrix for the fit points
        # V[i, j] = x_fit[i] ** j
        V = np.vander(x_fit, N=6, increasing=True)
        
        # Weighted least squares:
        # (V^T W V) c = V^T W y
        # W is diagonal with w_fit
        
        # Efficient computation:
        # V_w = V * sqrt(w_fit)[:, None]
        # y_w = y_fit * sqrt(w_fit)
        # c = lstsq(V_w, y_w)
        
        sqrt_w = np.sqrt(w_fit)
        V_w = V * sqrt_w[:, None]
        y_w = y_fit * sqrt_w
        
        # Use lstsq for stability
        coeff, residuals, rank, s = np.linalg.lstsq(V_w, y_w, rcond=None)
        
        # Ensure finite
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
            
    except Exception:
        # Fallback to standard Chebyshev fit on raw data if binned fit fails
        try:
            x_min = np.min(x)
            x_max = np.max(x)
            x_range = x_max - x_min
            if x_range < 1e-10:
                return np.array([np.mean(y)] + [0.0]*5)
            
            x_center = (x_min + x_max) / 2.0
            x_scale = x_range / 2.0
            z = (x - x_center) / x_scale
            
            c = np.polynomial.chebyshev.chebfit(z, y, 5)
            a = np.polynomial.chebyshev.cheb2poly(c)
            if len(a) < 6:
                a = np.pad(a, (0, 6 - len(a)))
            elif len(a) > 6:
                a = a[:6]
                
            b = np.zeros(6)
            for i in range(6):
                if a[i] == 0:
                    continue
                coeff_scale = a[i] / (x_scale ** i)
                for j in range(i + 1):
                    # binom(i, j)
                    binom_val = 1.0
                    j_temp = min(j, i - j)
                    for k in range(1, j_temp + 1):
                        binom_val *= (i - j_temp + k) / k
                    term = coeff_scale * binom_val * ((-x_center) ** (i - j))
                    b[j] += term
            return b
        except Exception:
            return np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
