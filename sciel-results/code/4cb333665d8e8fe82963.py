"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
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
    
    # Denoise y using a Gaussian kernel smoother in x-space
    # This captures the smooth underlying relationship while filtering noise
    def gaussian_smooth(x_vals, y_vals, bandwidth):
        """Apply Gaussian kernel smoothing to denoise y as function of x."""
        n_pts = len(x_vals)
        y_smooth = np.zeros(n_pts)
        for i in range(n_pts):
            dists = x_vals - x_vals[i]
            weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
            w_sum = np.sum(weights)
            if w_sum > 1e-12:
                y_smooth[i] = np.sum(weights * y_vals) / w_sum
            else:
                y_smooth[i] = y_vals[i]
        return y_smooth
    
    # Try different bandwidths for smoothing
    # Use quantile-based bandwidth selection
    sorted_x = np.sort(x_norm)
    median_gap = np.median(np.diff(sorted_x)) if n > 1 else 1.0
    bandwidths_to_try = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    # Split for model selection
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_size = max(20, n // 5)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train = x_norm[train_idx]
    y_train = y[train_idx]
    x_val = x_norm[val_idx]
    y_val = y[val_idx]
    
    def fit_ridge(x_fit, y_fit, degree, lam):
        """Fit polynomial of given degree with ridge regularization."""
        V = np.vander(x_fit, N=degree + 1, increasing=True)
        A = V.T @ V + lam * np.eye(degree + 1)
        b = V.T @ y_fit
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
    
    # Strategy: Denoise y, then fit polynomial
    # For each bandwidth, denoise and fit, evaluate on validation
    
    best_score = np.inf
    best_coeff_norm = None
    best_degree = 5
    best_lam = 1e-4
    best_bandwidth = 0.5
    
    degrees_to_try = [5, 4, 3, 2]
    lambdas_to_try = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for bw in bandwidths_to_try:
        if time.time() - t_start > deadline - 15:
            break
        
        # Denoise training data
        y_train_smooth = gaussian_smooth(x_train, y_train, bw)
        
        for degree in degrees_to_try:
            if time.time() - t_start > deadline - 15:
                break
            for lam in lambdas_to_try:
                if time.time() - t_start > deadline - 15:
                    break
                
                coeff_norm = fit_ridge(x_train, y_train_smooth, degree, lam)
                
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
                    best_bandwidth = bw
    
    # Also try without smoothing as baseline
    for degree in degrees_to_try:
        if time.time() - t_start > deadline - 15:
            break
        for lam in lambdas_to_try:
            if time.time() - t_start > deadline - 15:
                break
            
            coeff_norm = fit_ridge(x_train, y_train, degree, lam)
            
            if not np.all(np.isfinite(coeff_norm)):
                continue
            
            pred_val = poly_eval(coeff_norm, x_val)
            val_mse = mse(pred_val, y_val)
            
            if val_mse < best_score:
                best_score = val_mse
                best_coeff_norm = coeff_norm.copy()
                best_degree = degree
                best_lam = lam
                best_bandwidth = -1  # No smoothing
    
    if best_coeff_norm is None:
        # Fallback: simple linear fit
        best_coeff_norm = fit_ridge(x_train, y_train, 1, 1e-4)
        best_degree = 1
        best_bandwidth = -1
    
    # Refit on ALL data with best hyperparameters
    if time.time() - t_start < deadline - 10:
        if best_bandwidth > 0:
            y_all_smooth = gaussian_smooth(x_norm, y, best_bandwidth)
        else:
            y_all_smooth = y
        
        final_coeff_norm = fit_ridge(x_norm, y_all_smooth, best_degree, best_lam)
        if np.all(np.isfinite(final_coeff_norm)):
            best_coeff_norm = final_coeff_norm
    
    # Pad to 6 coefficients if needed
    if len(best_coeff_norm) < 6:
        temp = np.zeros(6)
        temp[:len(best_coeff_norm)] = best_coeff_norm
        best_coeff_norm = temp
    
    # Convert from normalized x to original x scale
    # x_norm = (x - x_mean) / x_std
    # f(x_norm) = sum(c_i * x_norm^i)
    # g(x) = f((x - x_mean)/x_std) = sum(c_i * ((x - x_mean)/x_std)^i)
    
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
# EVOLVE-BLOCK-END
