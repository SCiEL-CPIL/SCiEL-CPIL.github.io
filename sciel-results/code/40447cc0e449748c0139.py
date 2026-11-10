"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def _fit_poly_scaled(x, y, degree, lam):
    """
    Fit polynomial in scaled basis with Tikhonov regularization.
    Returns coefficients in scaled basis (ascending powers of x_scaled).
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    # Heavier regularization for higher-order terms
    reg_weights = np.array([1.0, 1.0, 50.0, 500.0, 5000.0, 50000.0])[:degree + 1]
    R = np.diag(lam * reg_weights)
    
    try:
        A = V.T @ V + R
        b = V.T @ y
        c_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            c_scaled = np.zeros(degree + 1)
    
    return c_scaled, x_mean, x_std


def _scaled_to_monomial(c_scaled, x_mean, x_std, degree):
    """Convert coefficients from scaled basis to monomial basis in original x."""
    coeff_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        if abs(c_scaled[j]) < 1e-15:
            continue
        scale_factor = c_scaled[j] / (x_std ** j)
        for k in range(j + 1):
            coeff_original[k] += scale_factor * comb(j, k) * ((-x_mean) ** (j - k))
    return coeff_original


def _l1_threshold_fit(x, y, degree, threshold, max_iter=50):
    """
    Iterative L1 thresholding (soft thresholding on scaled-basis coefficients).
    Returns monomial coefficients.
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    reg_weights = np.array([1.0, 1.0, 50.0, 500.0, 5000.0, 50000.0])[:degree + 1]
    
    # Initialize with unregularized least squares
    try:
        c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
    except Exception:
        c_scaled = np.zeros(degree + 1)
    
    for _ in range(max_iter):
        # Soft threshold
        for i in range(degree + 1):
            t = threshold * reg_weights[i]
            if c_scaled[i] > t:
                c_scaled[i] -= t
            elif c_scaled[i] < -t:
                c_scaled[i] += t
            else:
                c_scaled[i] = 0.0
        
        # Re-fit with zeroed coefficients kept
        active = np.abs(c_scaled) > threshold * 0.1
        if np.sum(active) <= degree + 1:
            if np.sum(active) >= 1:
                V_active = V[:, active]
                try:
                    c_active = np.linalg.lstsq(V_active, y, rcond=None)[0]
                except Exception:
                    c_active = np.zeros(np.sum(active))
                c_new = np.zeros(degree + 1)
                c_new[active] = c_active
                # Re-apply threshold
                for i in range(degree + 1):
                    t = threshold * reg_weights[i]
                    if c_new[i] > t:
                        c_new[i] -= t
                    elif c_new[i] < -t:
                        c_new[i] += t
                    else:
                        c_new[i] = 0.0
                if np.allclose(c_new, c_scaled, atol=threshold * 0.01):
                    c_scaled = c_new
                    break
                c_scaled = c_new
            else:
                c_scaled = np.zeros(degree + 1)
                break
    
    return _scaled_to_monomial(c_scaled, x_mean, x_std, degree)


def _coordinate_descent_l1(V, y, lam, reg_weights, max_iter=200, tol=1e-6):
    """
    Coordinate descent for L1-regularized least squares.
    Minimizes ||V c - y||^2 + lam * sum(reg_weights[i] * |c[i]|)
    Returns coefficients in scaled basis.
    """
    n, p = V.shape
    c = np.zeros(p)
    
    # Precompute column norms squared
    col_norm_sq = np.sum(V ** 2, axis=0)
    
    for iteration in range(max_iter):
        max_change = 0.0
        for i in range(p):
            # Compute residual without contribution of c[i]
            r = y - V @ c
            r += V[:, i] * c[i]
            
            # Coordinate update
            rho = np.dot(V[:, i], r)
            denom = col_norm_sq[i]
            if denom < 1e-12:
                continue
            
            # Soft threshold
            t = lam * reg_weights[i]
            if rho > t:
                c_new = (rho - t) / denom
            elif rho < -t:
                c_new = (rho + t) / denom
            else:
                c_new = 0.0
            
            change = abs(c_new - c[i])
            max_change = max(max_change, change)
            c[i] = c_new
        
        if max_change < tol:
            break
    
    return c


def _fit_l1_scaled(x, y, degree, lam):
    """
    Fit polynomial in scaled basis with L1 regularization via coordinate descent.
    Returns coefficients in scaled basis (ascending powers of x_scaled).
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=degree + 1, increasing=True)
    
    # Heavier regularization for higher-order terms
    reg_weights = np.array([1.0, 1.0, 50.0, 500.0, 5000.0, 50000.0])[:degree + 1]
    
    c_scaled = _coordinate_descent_l1(V, y, lam, reg_weights)
    
    return c_scaled, x_mean, x_std


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 120s limit
    
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter non-finite
    mask = np.isfinite(training_x) & np.isfinite(training_subgrid_tendency)
    x = training_x[mask]
    y = training_subgrid_tendency[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    if n < 10:
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, min(1, n-1))
            if not np.all(np.isfinite(coeff)):
                coeff[0] = np.mean(y)
            return coeff
        except Exception:
            return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    # Internal cross-validation
    split = int(n * 0.8)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    
    best_coeff_monomial = None
    best_val_mse = np.inf
    
    # Strategy 1: Tikhonov regularization with varying lambda
    lambdas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    for lam in lambdas:
        if time.time() > deadline - 15.0:
            break
        
        try:
            c_scaled, x_mean, x_std = _fit_poly_scaled(x_train, y_train, degree=5, lam=lam)
            
            if not np.all(np.isfinite(c_scaled)):
                continue
            
            x_val_scaled = (x_val - x_mean) / x_std
            V_val = np.vander(x_val_scaled, N=6, increasing=True)
            pred_val = V_val @ c_scaled
            val_mse = np.mean((pred_val - y_val) ** 2)
            
            if not np.isfinite(val_mse):
                continue
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_coeff_monomial = _scaled_to_monomial(c_scaled, x_mean, x_std, degree=5)
        except Exception:
            continue
    
    # Strategy 2: L1 regularization via coordinate descent
    if time.time() < deadline - 10.0:
        l1_lams = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        for lam in l1_lams:
            if time.time() > deadline - 10.0:
                break
            
            try:
                c_scaled, x_mean, x_std = _fit_l1_scaled(x_train, y_train, degree=5, lam=lam)
                
                if not np.all(np.isfinite(c_scaled)):
                    continue
                
                # Evaluate on validation
                x_val_scaled = (x_val - x_mean) / x_std
                V_val = np.vander(x_val_scaled, N=6, increasing=True)
                pred_val = V_val @ c_scaled
                val_mse = np.mean((pred_val - y_val) ** 2)
                
                if not np.isfinite(val_mse):
                    continue
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_coeff_monomial = _scaled_to_monomial(c_scaled, x_mean, x_std, degree=5)
            except Exception:
                continue
    
    # Strategy 3: Final polish - refit on ALL data with best lambda from ridge
    if time.time() < deadline - 5.0:
        # Try a few refined lambda values around the best ridge solution
        # First, find which ridge lambda was best by re-evaluating
        best_ridge_lam = None
        best_ridge_val_mse = np.inf
        for lam in lambdas:
            try:
                c_scaled, x_mean, x_std = _fit_poly_scaled(x_train, y_train, degree=5, lam=lam)
                if not np.all(np.isfinite(c_scaled)):
                    continue
                x_val_scaled = (x_val - x_mean) / x_std
                V_val = np.vander(x_val_scaled, N=6, increasing=True)
                pred_val = V_val @ c_scaled
                val_mse = np.mean((pred_val - y_val) ** 2)
                if np.isfinite(val_mse) and val_mse < best_ridge_val_mse:
                    best_ridge_val_mse = val_mse
                    best_ridge_lam = lam
            except Exception:
                continue
        
        if best_ridge_lam is not None:
            # Refit on all data with best lambda
            try:
                c_scaled_full, x_mean_full, x_std_full = _fit_poly_scaled(x, y, degree=5, lam=best_ridge_lam)
                if np.all(np.isfinite(c_scaled_full)):
                    coeff_full = _scaled_to_monomial(c_scaled_full, x_mean_full, x_std_full, degree=5)
                    if np.all(np.isfinite(coeff_full)):
                        # Use full-data fit as final answer (it should generalize similarly)
                        best_coeff_monomial = coeff_full
            except Exception:
                pass
    
    # Fallback to linear fit
    if best_coeff_monomial is None or not np.all(np.isfinite(best_coeff_monomial)):
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            if not np.all(np.isfinite(coeff)):
                coeff[0] = np.mean(y)
            return coeff
        except Exception:
            return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    if not np.all(np.isfinite(best_coeff_monomial)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff[0] = np.mean(y)
        return coeff
    
    # Final sanity check: ensure coefficients are finite and not absurdly large
    if not np.all(np.isfinite(best_coeff_monomial)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff[0] = np.mean(y)
        return coeff
    
    # Clamp extremely large coefficients that might indicate numerical issues
    max_coeff = np.max(np.abs(best_coeff_monomial))
    if max_coeff > 1e10:
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff[0] = np.mean(y)
        return coeff
    
    return best_coeff_monomial
# EVOLVE-BLOCK-END
