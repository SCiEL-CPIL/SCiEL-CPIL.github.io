"""Adaptive-degree polynomial closure via K-Fold Cross-Validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 100.0  # Safety margin before 120s limit

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

    # Normalize x for numerical stability in fitting
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    x_norm = (x - x_mean) / x_std

    # K-Fold Cross-Validation to select best degree (0 to 5)
    # We use 5 folds for a reasonable balance of bias/variance in selection
    k_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, k_folds)

    best_cv_mse = np.inf
    best_degree = 0
    best_coeff_norm = np.zeros(6)

    # Try degrees 0 through 5
    for deg in range(6):
        if time.time() - t_start > deadline:
            break
            
        cv_mses = []
        for i in range(k_folds):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
            
            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]
            
            # Build Vandermonde for training
            V_train = np.vander(x_train, deg + 1, increasing=True)
            
            # Solve least squares
            try:
                coeff, residuals, rank, sv = np.linalg.lstsq(V_train, y_train, rcond=None)
            except Exception:
                continue
                
            # Evaluate on validation
            V_val = np.vander(x_val, deg + 1, increasing=True)
            y_pred = V_val @ coeff
            mse = np.mean((y_pred - y_val) ** 2)
            cv_mses.append(mse)
            
        if cv_mses:
            mean_cv_mse = np.mean(cv_mses)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_degree = deg
                # Store the coefficients from the last fold? No, we need to refit on all data
                # But we can store the 'shape' of the solution. Actually, we just need to refit.
                # We don't store coeff here because it's from a subset.
                pass

    # Refit on all data with the best degree
    if time.time() - t_start > deadline:
        # Fallback to simple mean if running out of time
        return np.array([np.mean(y)] + [0.0]*5)

    V_all = np.vander(x_norm, best_degree + 1, increasing=True)
    try:
        coeff_norm, residuals, rank, sv = np.linalg.lstsq(V_all, y, rcond=None)
    except Exception:
        # Fallback to constant
        coeff_norm = np.zeros(6)
        coeff_norm[0] = np.mean(y)

    # Convert normalized coefficients (in terms of t = (x - x_mean)/x_std)
    # to original x coefficients (in terms of x)
    # f(x) = sum_{k=0}^{best_degree} coeff_norm[k] * ((x - x_mean)/x_std)^k
    # Let t = (1/x_std)*x - x_mean/x_std = a*x + b
    # t^k = (a*x + b)^k = sum_{j=0}^{k} binom(k, j) * a^j * b^(k-j) * x^j
    # c_j = sum_{k=j}^{best_degree} coeff_norm[k] * binom(k, j) * a^j * b^(k-j)
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    c = np.zeros(6)
    for k in range(best_degree + 1):
        ck = coeff_norm[k]
        if ck == 0:
            continue
        for j in range(k + 1):
            c[j] += ck * comb(k, j) * (a ** j) * (b ** (k - j))

    return c
# EVOLVE-BLOCK-END
