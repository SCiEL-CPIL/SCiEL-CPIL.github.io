import numpy as np
import time
from scipy import stats

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
    
    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_norm = (x - x_mean) / x_std
    
    # Split for model selection
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_size = max(100, n // 10)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train = x_norm[train_idx]
    y_train = y[train_idx]
    x_val = x_norm[val_idx]
    y_val = y[val_idx]
    
    def fit_wls(x_fit, y_fit, weights, degree, lam):
        """Fit polynomial of given degree with weighted ridge regularization."""
        V = np.vander(x_fit, N=degree + 1, increasing=True)
        W = np.diag(weights)
        A = V.T @ W @ V + lam * np.eye(degree + 1)
        b = V.T @ W @ y_fit
        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                return np.zeros(degree + 1)
        return coeff
    
    def poly_eval(coeff, x_eval):
        """Evaluate polynomial with given coefficients."""
        pred = np.zeros_like(x_eval)
        for i in range(len(coeff)):
            pred += coeff[i] * (x_eval ** i)
        return pred
    
    def mse(pred, target):
        return np.mean((pred - target) ** 2)
    
    best_score = np.inf
    best_coeff_norm = None
    best_degree = 5
    best_lam = 1e-4
    best_n_bins = 100
    
    degrees_to_try = [5, 4, 3]
    lambdas_to_try = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    n_bins_to_try = [50, 100, 200, 500]
    
    # Search over hyperparameters
    for n_bins in n_bins_to_try:
        if time.time() - t_start > deadline - 10:
            break
        
        # Bin the training data
        bin_edges = np.linspace(np.min(x_train), np.max(x_train), n_bins + 1)
        bin_indices = np.digitize(x_train, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute binned means and weights
        binned_y = np.zeros(n_bins)
        weights = np.zeros(n_bins)
        for i in range(n_bins):
            mask_bin = (bin_indices == i)
            if np.sum(mask_bin) > 0:
                binned_y[i] = np.mean(y_train[mask_bin])
                weights[i] = np.sum(mask_bin)
            else:
                binned_y[i] = 0.0
                weights[i] = 0.0
        
        # Only use bins with at least one point
        valid_bins = weights > 0
        x_binned = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        x_binned = x_binned[valid_bins]
        y_binned = binned_y[valid_bins]
        weights_binned = weights[valid_bins]
        
        if len(x_binned) < degree + 1:
            continue
        
        for degree in degrees_to_try:
            if time.time() - t_start > deadline - 10:
                break
            for lam in lambdas_to_try:
                if time.time() - t_start > deadline - 10:
                    break
                
                coeff_norm = fit_wls(x_binned, y_binned, weights_binned, degree, lam)
                
                if not np.all(np.isfinite(coeff_norm)):
                    continue
                
                # Evaluate on validation (use raw y_val for fair comparison)
                pred_val = poly_eval(coeff_norm, x_val)
                val_mse = mse(pred_val, y_val)
                
                if val_mse < best_score:
                    best_score = val_mse
                    best_coeff_norm = coeff_norm.copy()
                    best_degree = degree
                    best_lam = lam
                    best_n_bins = n_bins
    
    if best_coeff_norm is None:
        # Fallback: simple linear fit
        best_coeff_norm = fit_wls(x_train, y_train, np.ones_like(x_train), 1, 1e-4)
        best_degree = 1
        best_n_bins = 1
    
    # Refit on ALL data with best hyperparameters
    if time.time() - t_start < deadline - 5:
        n_bins = best_n_bins
        bin_edges = np.linspace(np.min(x_norm), np.max(x_norm), n_bins + 1)
        bin_indices = np.digitize(x_norm, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        binned_y = np.zeros(n_bins)
        weights = np.zeros(n_bins)
        for i in range(n_bins):
            mask_bin = (bin_indices == i)
            if np.sum(mask_bin) > 0:
                binned_y[i] = np.mean(y[mask_bin])
                weights[i] = np.sum(mask_bin)
            else:
                binned_y[i] = 0.0
                weights[i] = 0.0
        
        valid_bins = weights > 0
        x_binned = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        x_binned = x_binned[valid_bins]
        y_binned = binned_y[valid_bins]
        weights_binned = weights[valid_bins]
        
        if len(x_binned) >= best_degree + 1:
            final_coeff_norm = fit_wls(x_binned, y_binned, weights_binned, best_degree, best_lam)
            if np.all(np.isfinite(final_coeff_norm)):
                best_coeff_norm = final_coeff_norm
    
    # Pad to 6 coefficients if needed
    if len(best_coeff_norm) < 6:
        temp = np.zeros(6)
        temp[:len(best_coeff_norm)] = best_coeff_norm
        best_coeff_norm = temp
    
    # Convert from normalized x to original x scale
    max_deg = min(5, len(best_coeff_norm) - 1)
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for i in range(max_deg + 1):
        if abs(best_coeff_norm[i]) < 1e-15:
            continue
        
        ci_over_std_pow = best_coeff_norm[i] / (x_std ** i)
        
        # Expand (x - x_mean)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
        for j in range(i + 1):
            # Compute binomial coefficient
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
            
            term = binom_val * ((-x_mean) ** (i - j)) * ci_over_std_pow
            coeff_orig[j] += term
    
    # Ensure finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
    
    return coeff_orig
