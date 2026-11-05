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
    
    # Try multiple regularization strengths and pick the best
    # Use a simple approach: fit with different lambda values and select based on
    # an internal validation split
    
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
    
    # K-fold cross-validation for lambda selection
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    
    # Use 5-fold CV if enough data, otherwise fewer folds
    if n >= 30:
        n_folds = 5
    elif n >= 12:
        n_folds = 3
    else:
        n_folds = 2
    
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        folds.append(indices[start:end])
    
    # Regularization: penalize higher-order terms more heavily
    reg_weights = np.array([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    reg_matrix = np.diag(reg_weights)
    
    # Log-spaced regularization values - broader range for better exploration
    lambdas = [0.0] + list(np.logspace(-15, 6, 40))
    
    best_mse = np.inf
    best_coeffs_norm = None
    
    # Precompute V_train and y_train for each fold to avoid repeated slicing
    fold_train_data = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        fold_train_data.append((V_norm[train_idx], y[train_idx], V_norm[val_idx], y[val_idx]))
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
        try:
            mse_sum = 0.0
            count = 0
            valid_fold = True
            for f in range(n_folds):
                V_train, y_train, V_val, y_val = fold_train_data[f]
                
                if lam == 0.0:
                    coeffs_lam, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
                else:
                    VtV = V_train.T @ V_train
                    Vty = V_train.T @ y_train
                    A = VtV + lam * reg_matrix
                    try:
                        coeffs_lam = np.linalg.solve(A, Vty)
                    except np.linalg.LinAlgError:
                        coeffs_lam, _, _, _ = np.linalg.lstsq(A, Vty, rcond=None)
                
                if not np.all(np.isfinite(coeffs_lam)):
                    valid_fold = False
                    break
                
                y_val_pred = V_val @ coeffs_lam
                mse_sum += np.mean((y_val_pred - y_val) ** 2)
                count += 1
            
            if not valid_fold or count != n_folds:
                continue
            
            avg_mse = mse_sum / n_folds
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_coeffs_norm = coeffs_lam.copy()
        except:
            continue
    
    if best_coeffs_norm is None:
        # Fallback to OLS on all data
        try:
            coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if np.all(np.isfinite(coeffs_norm)):
                best_coeffs_norm = coeffs_norm
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert normalized coefficients to original x-space using exact binomial expansion
    # P(x) = sum_{k=0}^5 c_k * ((x - x_mean) / x_std)^k
    #      = sum_{k=0}^5 c_k * x_std^{-k} * (x - x_mean)^k
    #      = sum_{k=0}^5 c_k * x_std^{-k} * sum_{j=0}^k binom(k, j) * x^j * (-x_mean)^(k-j)
    #      = sum_{j=0}^5 [ sum_{k=j}^5 c_k * x_std^{-k} * binom(k, j) * (-x_mean)^(k-j) ] * x^j
    
    coeff_orig = np.zeros(6)
    for k in range(6):
        c_k = best_coeffs_norm[k]
        if not np.isfinite(c_k):
            continue
        inv_std_k = 1.0 / (x_std ** k)
        for j in range(k + 1):
            coeff_orig[j] += c_k * inv_std_k * comb(k, j) * ((-x_mean) ** (k - j))
    
    # Final polish: refine coefficients with a local optimization pass
    # Use the CV-selected coefficients as initial guess, then do a fine-grained
    # search around them to reduce residual error
    if np.all(np.isfinite(coeff_orig)) and n > 6:
        # Try a small local refinement in the original coefficient space
        # by evaluating on training data and adjusting if it improves fit
        x_test = x[:min(100, n)]
        y_test = y[:min(100, n)]
        
        # Build Vandermonde in original space for testing
        V_orig = np.zeros((len(x_test), 6))
        for i in range(6):
            V_orig[:, i] = x_test ** i
        
        # Current MSE
        pred = V_orig @ coeff_orig
        current_mse = np.mean((pred - y_test) ** 2)
        
        # Try small perturbations to see if we can improve
        # This is a heuristic refinement step
        best_local = coeff_orig.copy()
        best_local_mse = current_mse
        
        # Only do this if we have time
        if time.time() - start_time < deadline - 5:
            # Try adjusting each coefficient slightly
            for i in range(6):
                for delta_factor in [0.9, 0.95, 1.05, 1.1]:
                    trial = coeff_orig.copy()
                    trial[i] *= delta_factor
                    if not np.all(np.isfinite(trial)):
                        continue
                    pred_trial = V_orig @ trial
                    mse_trial = np.mean((pred_trial - y_test) ** 2)
                    if mse_trial < best_local_mse:
                        best_local_mse = mse_trial
                        best_local = trial.copy()
        
        if np.all(np.isfinite(best_local)):
            coeff_orig = best_local
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
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
