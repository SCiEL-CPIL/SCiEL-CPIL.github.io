"""Rewritten program using MODEL-AND-OPTIMIZE: fit full degree-5 polynomial with regularization."""
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
    
    # First, normalize x to improve numerical conditioning
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
            # Pad if necessary (though lstsq returns 6 coeffs if V has 6 cols)
            if len(coeffs_norm) < 6:
                tmp = np.zeros(6)
                tmp[:len(coeffs_norm)] = coeffs_norm
                coeffs_norm = tmp
            # Convert back to original scale
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
    
    # Use K-Fold Cross-Validation to select regularization parameter
    # This is more robust than a single split.
    # We will test a range of lambda values.
    # To save time, we pre-compute VtV and Vty for the full dataset? 
    # No, CV requires splitting. 
    # However, for Ridge Regression, we can use the closed form solution.
    
    # Strategy: 
    # 1. Define a grid of lambda values.
    # 2. For each lambda, perform K-Fold CV to estimate generalization error.
    # 3. Select the lambda with the lowest CV error.
    # 4. Refit on the entire dataset with the selected lambda.
    
    # Grid of lambdas
    # Log-spaced from very small to large
    # Note: lambda=0 is OLS. We handle it separately or as limit.
    lambdas = np.logspace(-10, 5, 50)
    lambdas = np.insert(lambdas, 0, 0.0)
    
    # K-Fold setup
    K = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    folds = np.array_split(indices, K)
    
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
            
            V_train = V_norm[train_indices]
            y_train = y[train_indices]
            V_val = V_norm[val_idx]
            y_val = y[val_idx]
            
            try:
                if lam == 0.0:
                    coeffs_lam, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
                else:
                    VtV = V_train.T @ V_train
                    Vty = V_train.T @ y_train
                    A = VtV + lam * np.eye(6)
                    coeffs_lam = np.linalg.solve(A, Vty)
                
                if not np.all(np.isfinite(coeffs_lam)):
                    valid_fold = False
                    break
                
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
            VtV = V_norm.T @ V_norm
            Vty = V_norm.T @ y
            A = VtV + best_lambda * np.eye(6)
            best_coeffs_norm = np.linalg.solve(A, Vty)
        
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
    # y = sum_{i=0}^{5} c_i * ((x - mean)/std)^i
    #     = sum_{i=0}^{5} (c_i / std^i) * (x - mean)^i
    # Expand (x - mean)^i in powers of x
    coeff_orig = np.zeros(6)
    for i in range(6):
        ci_scaled = best_coeffs_norm[i] / (x_std ** i)
        # (x - x_mean)^i = sum_{k=0}^{i} binom(i,k) x^k (-x_mean)^{i-k}
        for k in range(i + 1):
            coeff_orig[k] += ci_scaled * comb(i, k) * ((-x_mean) ** (i - k))
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback to linear fit
        try:
            c_lin = np.polyfit(x, y, 1)  # descending powers
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]  # constant
            coeff_orig[1] = c_lin[0]  # linear
        except:
            coeff_orig = np.zeros(6)
    
    # Final check: all must be finite
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
