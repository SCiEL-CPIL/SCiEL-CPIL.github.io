import numpy as np
import time
import math

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
    
    # Pre-compute powers of u up to degree 5
    # Shape: (n, 6)
    u_powers = np.ones((n, 6))
    for i in range(1, 6):
        u_powers[:, i] = u_powers[:, i-1] * u

    # 2. K-Fold Cross-Validation
    K = 5
    if n > 500:
        K = 10
    
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[:n % K] += 1
    folds = np.array_split(indices, K)

    degrees = [0, 1, 2, 3, 4, 5]
    alphas = np.logspace(-6, 6, 13)

    best_cv_err = np.inf
    best_deg = 1
    best_alpha = 1.0

    for deg in degrees:
        if time.time() > deadline - 10.0:
            break
            
        min_err_for_deg = np.inf
        best_alpha_for_deg = 1.0
        
        for alpha in alphas:
            if time.time() > deadline - 10.0:
                break
                
            cv_errors = []
            
            for i in range(K):
                if time.time() > deadline - 10.0:
                    break
                    
                test_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
                
                A_train = u_powers[train_idx, :deg+1]
                A_test = u_powers[test_idx, :deg+1]
                
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                y_mean = np.mean(y_train_fold)
                y_train_dm = y_train_fold - y_mean
                
                try:
                    ATA = A_train.T @ A_train
                    ATy = A_train.T @ y_train_dm
                    
                    reg = alpha * np.eye(deg + 1)
                    reg[0, 0] = 0.0 # No penalty on intercept
                    
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

    # 3. Final Fit
    A_all = u_powers[:, :best_deg+1]
    y_mean_all = np.mean(y)
    y_all_dm = y - y_mean_all
    
    ATA = A_all.T @ A_all
    ATy = A_all.T @ y_all_dm
    
    reg = best_alpha * np.eye(best_deg + 1)
    reg[0, 0] = 0.0
    
    try:
        coeff_dm = np.linalg.solve(ATA + reg, ATy)
    except np.linalg.LinAlgError:
        coeff_dm = np.linalg.lstsq(A_all, y_all_dm, rcond=None)[0]
        
    # 4. Convert coefficients from u-basis to x-basis
    c_x = np.zeros(6)
    
    for i in range(best_deg + 1):
        a_i = coeff_dm[i]
        if a_i == 0:
            continue
        scale_i = a_i / (sigma ** i)
        
        for k in range(i + 1):
            binom_val = math.comb(i, k)
            term_coeff = binom_val * ((-mu) ** (i - k))
            c_x[k] += scale_i * term_coeff
            
    if not np.all(np.isfinite(c_x)):
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            c_x = np.zeros(6)
            c_x[:2] = c_lin
        except Exception:
            c_x = np.zeros(6)

    return c_x
