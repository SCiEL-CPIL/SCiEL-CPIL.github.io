import numpy as np
import time
import math
from numpy.polynomial import polynomial as P

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6)

    n = x.size

    if n < 10:
        # Fallback for very small data
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff = np.zeros(6)
            coeff[:2] = c_lin
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # 1. Scale x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
    u = (x - mu) / sigma
    u_train = u

    # 2. K-Fold Cross-Validation to select degree and lambda
    # Use stratified or random split? Random is fine for regression.
    # K=5 or 10. Let's use 5 for speed, or 10 if n is large.
    K = 5
    if n > 500:
        K = 10
    
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[:n % K] += 1
    folds = np.array_split(indices, K)

    degrees = [0, 1, 2, 3, 4, 5]
    # Regularization strengths (Ridge alpha)
    # Since we scaled u to variance 1, alpha is in units of (y^2 / x^2) roughly.
    # Let's try a wide range.
    alphas = np.logspace(-6, 6, 13)

    best_cv_err = np.inf
    best_deg = 1
    best_alpha = 1.0
    
    for deg in degrees:
        if time.time() > deadline - 10.0:
            break
            
        best_alpha_for_deg = 1.0
        min_err_for_deg = np.inf
        
        for alpha in alphas:
            if time.time() > deadline - 10.0:
                break
                
            cv_errors = []
            
            for i in range(K):
                test_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
                
                u_train_fold = u_train[train_idx]
                y_train_fold = y[train_idx]
                u_test_fold = u_train[test_idx]
                y_test_fold = y[test_idx]
                
                A_train = np.column_stack([u_train_fold**i for i in range(deg + 1)])
                A_test = np.column_stack([u_test_fold**i for i in range(deg + 1)])
                
                y_mean = np.mean(y_train_fold)
                y_train_dm = y_train_fold - y_mean
                
                try:
                    ATA = A_train.T @ A_train
                    ATy = A_train.T @ y_train_dm
                    
                    reg = alpha * np.eye(deg + 1)
                    reg[0, 0] = 0.0
                    
                    coeff_dm = np.linalg.solve(ATA + reg, ATy)
                    
                    pred_dm_test = A_test @ coeff_dm
                    pred_test = pred_dm_test + y_mean
                    
                    err = np.mean((pred_test - y_test_fold)**2)
                    cv_errors.append(err)
                    
                except np.linalg.LinAlgError:
                    continue

            if cv_errors:
                mean_err = np.mean(cv_errors)
                if mean_err < min_err_for_deg:
                    min_err_for_deg = mean_err
                    best_alpha_for_deg = alpha
                    
        if min_err_for_deg < best_cv_err:
            best_cv_err = min_err_for_deg
            best_deg = deg
            best_alpha = best_alpha_for_deg

    # 3. Final Fit on all training data with best_deg and best_alpha
    u_all = u_train
    A_all = np.column_stack([u_all**i for i in range(best_deg + 1)])
    
    y_mean_all = np.mean(y)
    y_all_dm = y - y_mean_all
    
    ATA = A_all.T @ A_all
    ATy = A_all.T @ y_all_dm
    
    reg = best_alpha * np.eye(best_deg + 1)
    reg[0, 0] = 0.0
    
    try:
        coeff_dm = np.linalg.solve(ATA + reg, ATy)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (shouldn't happen with ridge)
        coeff_dm = np.linalg.lstsq(A_all, y_all_dm, rcond=None)[0]
        
    # coeff_dm are coefficients for u^0 ... u^deg
    # We need to convert to x^0 ... x^5
    
    # u = (x - mu) / sigma
    # u^i = (1/sigma^i) * (x - mu)^i
    # (x - mu)^i = sum_{k=0}^i binom(i, k) x^k (-mu)^(i-k)
    
    c_x = np.zeros(6)
    
    for i in range(best_deg + 1):
        a_i = coeff_dm[i]
        scale_i = a_i / (sigma ** i)
        
        # Expand (x - mu)^i
        for k in range(i + 1):
            # Term x^k
            # Coeff is binom(i, k) * (-mu)^(i-k)
            binom_val = math.comb(i, k)
            
            term_coeff = binom_val * ((-mu) ** (i - k))
            c_x[k] += scale_i * term_coeff
            
    # Ensure finite
    if not np.all(np.isfinite(c_x)):
        # Fallback to simple linear fit on original scale
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            c_x = np.zeros(6)
            c_x[:2] = c_lin
        except Exception:
            c_x = np.zeros(6)

    return c_x
