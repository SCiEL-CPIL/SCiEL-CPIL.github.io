"""Strategy: Weighted polynomial fit on binned means with regularization and cross-validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb

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

    # Check for constant x
    x_range = np.ptp(x)
    if x_range < 1e-12:
        coeff = np.zeros(6)
        coeff[0] = np.mean(y)
        return coeff

    # Standardization parameters
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x < 1e-12:
        sigma_x = 1.0

    def convert_coefficients(d, mu, sigma):
        """
        Convert coefficients d of polynomial in z=(x-mu)/sigma to coefficients c of polynomial in x.
        p(x) = sum_{k=0}^5 d_k ((x-mu)/sigma)^k
             = sum_{k=0}^5 d_k sigma^{-k} (x-mu)^k
             = sum_{j=0}^5 x^j [ sum_{k=j}^5 d_k sigma^{-k} C(k,j) (-mu)^{k-j} ]
        """
        c = np.zeros(6)
        for j in range(6):
            c_j = 0.0
            for k in range(j, 6):
                c_j += d[k] * (sigma ** -k) * comb(k, j) * ((-mu) ** (k - j))
            c[j] = c_j
        return c

    def fit_model(x_train, y_train, n_bins, lam):
        """
        Fit polynomial to binned means of training data.
        Returns coefficients in original x space.
        """
        # Scale x
        z_train = (x_train - mu_x) / sigma_x
        
        # Bin z_train
        # Use equal-width bins covering the range of z_train
        z_min = np.min(z_train)
        z_max = np.max(z_train)
        if z_max - z_min < 1e-12:
            # Fallback to mean if range is tiny
            return np.array([np.mean(y_train)] + [0.0]*5)
        
        # Create bins
        bin_edges = np.linspace(z_min, z_max, n_bins + 1)
        # Assign each point to a bin
        bin_indices = np.digitize(z_train, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute bin means and counts
        bin_counts = np.zeros(n_bins)
        bin_sums = np.zeros(n_bins)
        
        np.add.at(bin_counts, bin_indices, 1)
        np.add.at(bin_sums, bin_indices, y_train)
        
        # Identify non-empty bins
        non_empty = bin_counts > 0
        if np.sum(non_empty) < 6:
            # Not enough data points for a degree 5 fit
            # Fall back to a lower degree or just mean
            # For simplicity, return mean-based coeff or try lower degree
            # But we must return degree 5.
            # Let's just return a very regularized fit or mean
            # Using a large lambda effectively does this
            pass
            
        z_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        y_means = np.zeros(n_bins)
        y_means[non_empty] = bin_sums[non_empty] / bin_counts[non_empty]
        
        # Only use non-empty bins for fitting
        z_fit = z_centers[non_empty]
        y_fit = y_means[non_empty]
        w_fit = bin_counts[non_empty]
        
        if len(z_fit) < 6:
             # Fallback: fit to raw data with strong regularization
             # Or just return mean
             # Let's try to fit raw data with huge lambda
             z_raw = (x_train - mu_x) / sigma_x
             V_raw = np.vander(z_raw, 6, increasing=True)
             # Huge regularization
             A = V_raw.T @ V_raw + 1e10 * np.eye(6)
             b = V_raw.T @ y_train
             try:
                 d = np.linalg.solve(A, b)
                 return convert_coefficients(d, mu_x, sigma_x)
             except:
                 return np.array([np.mean(y_train)] + [0.0]*5)

        # Construct Vandermonde matrix for bin centers
        V = np.vander(z_fit, 6, increasing=True)
        
        # Weighted Normal Equations
        # A = V^T W V + lam * I
        # b = V^T W y
        # W is diagonal with w_fit
        # V^T W V = sum_i w_i V_i V_i^T
        # Efficient computation:
        VW = V * w_fit[:, np.newaxis] # Scale rows
        A = VW.T @ V + lam * np.eye(6)
        b = VW.T @ y_fit
        
        try:
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If singular, use pinv
            d = np.linalg.pinv(A) @ b
            
        return convert_coefficients(d, mu_x, sigma_x)

    # Cross-Validation
    n_folds = 5
    fold_size = n // n_folds
    
    # Candidate parameters
    n_bins_list = [10, 20, 50, 100, 200]
    lam_list = [0.0, 1e-6, 1e-4, 1e-2, 1e-1]
    
    best_cv_error = np.inf
    best_params = (50, 1e-4) # Default
    
    # Shuffle indices for CV
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    for n_bins in n_bins_list:
        if time.time() > deadline - 20:
            break
        for lam in lam_list:
            if time.time() > deadline - 10:
                break
                
            cv_errors = []
            for fold in range(n_folds):
                # Define validation and training sets
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else n
                
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                x_train = x[train_idx]
                y_train = y[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]
                
                # Fit on training
                coeff = fit_model(x_train, y_train, n_bins, lam)
                
                # Evaluate on validation
                # pred = sum(c_i * x_val**i)
                pred = np.polynomial.polynomial.polyval(x_val, coeff)
                mse = np.mean((pred - y_val) ** 2)
                cv_errors.append(mse)
            
            if cv_errors:
                mean_cv_error = np.mean(cv_errors)
                if mean_cv_error < best_cv_error:
                    best_cv_error = mean_cv_error
                    best_params = (n_bins, lam)
    
    # Final fit with best parameters
    best_n_bins, best_lam = best_params
    final_coeff = fit_model(x, y, best_n_bins, best_lam)
    
    return final_coeff

# EVOLVE-BLOCK-END
