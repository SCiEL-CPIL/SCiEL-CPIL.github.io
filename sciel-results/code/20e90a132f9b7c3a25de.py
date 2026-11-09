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
    
    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_norm = (x - x_mean) / x_std
    
    # Split into train and validation for model selection
    if n > 200:
        np.random.seed(42)
        indices = np.random.permutation(n)
        val_size = max(50, n // 5)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        x_train = x_norm[train_idx]
        y_train = y[train_idx]
        x_val = x_norm[val_idx]
        y_val = y[val_idx]
    else:
        x_train = x_norm
        y_train = y
        x_val = x_norm
        y_val = y
    
    def fit_ridge(x_fit, y_fit, degree, lam):
        """Fit polynomial of given degree with ridge regularization in normalized space."""
        V = np.vander(x_fit, N=degree + 1, increasing=True)
        
        # Ridge regression
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
    
    def poly_eval_horner(coeff, x_eval):
        """Evaluate polynomial using Horner's method for numerical stability."""
        pred = np.zeros_like(x_eval)
        for i in range(len(coeff) - 1, -1, -1):
            pred = pred * x_eval + coeff[i]
        return pred
    
    def mse(pred, target):
        return np.mean((pred - target) ** 2)
    
    # Try different degrees and regularization strengths
    # Finer lambda grid for better selection
    best_score = np.inf
    best_coeff_norm = None
    best_degree = 5
    best_lam = 1e-4
    
    degrees_to_try = [5, 4, 3, 2, 1]
    # Finer grid, especially around the likely optimal range
    lambdas_to_try = np.logspace(-12, 2, 30)
    
    for degree in degrees_to_try:
        if time.time() - t_start > deadline - 15:
            break
        for lam in lambdas_to_try:
            if time.time() - t_start > deadline - 15:
                break
            
            coeff_norm = fit_ridge(x_train, y_train, degree, lam)
            
            if not np.all(np.isfinite(coeff_norm)):
                continue
            
            # Evaluate on validation using Horner's method
            pred_val = poly_eval_horner(coeff_norm, x_val)
            val_mse = mse(pred_val, y_val)
            
            if val_mse < best_score:
                best_score = val_mse
                best_coeff_norm = coeff_norm.copy()
                best_degree = degree
                best_lam = lam
    
    if best_coeff_norm is None:
        # Fallback: simple linear fit
        best_coeff_norm = fit_ridge(x_train, y_train, 1, 1e-4)
        best_degree = 1
    
    # Now, refit on ALL data with the best lambda and degree
    if time.time() - t_start < deadline - 10:
        final_coeff_norm = fit_ridge(x_norm, y, best_degree, best_lam)
        if np.all(np.isfinite(final_coeff_norm)):
            best_coeff_norm = final_coeff_norm
    
    # Pad to 6 coefficients if needed
    if len(best_coeff_norm) < 6:
        temp = np.zeros(6)
        temp[:len(best_coeff_norm)] = best_coeff_norm
        best_coeff_norm = temp
    
    # Convert from normalized x to original x scale
    # x_norm = (x - x_mean) / x_std
    # We have f(x_norm) = sum(c_i * x_norm^i)
    # Need g(x) = f((x - x_mean)/x_std) = sum(c_i * ((x - x_mean)/x_std)^i)
    
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
