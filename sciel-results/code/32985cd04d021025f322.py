"""Combined program: K-fold CV for lambda selection with refined grid and efficient computation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (and 150s kill)
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n == 0:
        return np.zeros(6)
    
    # Normalize x to improve numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for normalized x
    def build_vandermonde(xn, degree=5):
        """Build Vandermonde matrix for polynomial fit in normalized coords."""
        V = np.zeros((len(xn), degree + 1))
        for i in range(degree + 1):
            V[:, i] = xn ** i
        return V
    
    V_norm = build_vandermonde(x_norm, 5)
    
    if n < 6:
        # Not enough data for degree 5, fallback to linear or zero
        try:
            coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if len(coeffs_norm) < 6:
                tmp = np.zeros(6)
                tmp[:len(coeffs_norm)] = coeffs_norm
                coeffs_norm = tmp
            coeff_orig = np.zeros(6)
            for i in range(6):
                ci = coeffs_norm[i] / (x_std ** i)
                for k in range(i + 1):
                    coeff_orig[k] += ci * comb(i, k) * ((-x_mean) ** (i - k))
            if np.all(np.isfinite(coeff_orig)):
                return coeff_orig
        except:
            pass
        return np.zeros(6)
    
    # K-Fold Cross-Validation to select regularization parameter
    # Use a refined grid of lambda values with more points in the critical range
    lambdas = np.logspace(-12, 8, 60)
    lambdas = np.insert(lambdas, 0, 0.0)
    
    # K-Fold setup - use more folds for better estimation when data allows
    K = min(10, max(5, n // 500))  # Scale folds with data size
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    folds = np.array_split(indices, K)
    
    # Pre-compute VtV and Vty for the full dataset to speed up ridge solutions
    # For each fold, we need V_train^T V_train and V_train^T y_train
    # We can compute these efficiently using the full VtV and subtracting the validation fold
    
    VtV_full = V_norm.T @ V_norm
    Vty_full = V_norm.T @ y
    
    best_cv_mse = np.inf
    best_lambda = 0.0
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
            
        total_mse = 0.0
        valid_fold = True
        
        for i in range(K):
            val_idx = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(K) if j != i])
            
            # Compute VtV and Vty for training set using subtraction
            VtV_train = VtV_full - V_norm[val_idx].T @ V_norm[val_idx]
            Vty_train = Vty_full - V_norm[val_idx].T @ y[val_idx]
            
            try:
                if lam == 0.0:
                    # For OLS, use lstsq on the training data
                    V_train = V_norm[train_indices]
                    y_train = y[train_indices]
                    coeffs_lam, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
                else:
                    A = VtV_train + lam * np.eye(6)
                    coeffs_lam = np.linalg.solve(A, Vty_train)
                
                if not np.all(np.isfinite(coeffs_lam)):
                    valid_fold = False
                    break
                
                V_val = V_norm[val_idx]
                y_val = y[val_idx]
                y_val_pred = V_val @ coeffs_lam
                mse_val = np.mean((y_val_pred - y_val) ** 2)
                total_mse += mse_val
            except:
                valid_fold = False
                break
        
        if not valid_fold:
            continue
            
        avg_cv_mse = total_mse / K
        
        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_lambda = lam
    
    # Refit with the best lambda on all data
    try:
        if best_lambda == 0.0:
            best_coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
        else:
            A = VtV_full + best_lambda * np.eye(6)
            best_coeffs_norm = np.linalg.solve(A, Vty_full)
        
        if not np.all(np.isfinite(best_coeffs_norm)):
            # Fallback to OLS
            best_coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if not np.all(np.isfinite(best_coeffs_norm)):
                return np.zeros(6)
    except:
        # Fallback to OLS
        try:
            best_coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if not np.all(np.isfinite(best_coeffs_norm)):
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert normalized coefficients back to original x scale
    coeff_orig = np.zeros(6)
    for i in range(6):
        ci_scaled = best_coeffs_norm[i] / (x_std ** i)
        for k in range(i + 1):
            coeff_orig[k] += ci_scaled * comb(i, k) * ((-x_mean) ** (i - k))
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback to linear fit
        try:
            c_lin = np.polyfit(x, y, 1)
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]
            coeff_orig[1] = c_lin[0]
        except:
            coeff_orig = np.zeros(6)
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
