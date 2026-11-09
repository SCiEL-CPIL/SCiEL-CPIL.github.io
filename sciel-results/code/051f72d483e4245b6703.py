import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
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
    
    # We want to fit a degree-5 polynomial: sum(c_i * x^i)
    # Using standardized x: sum(a_i * x_norm^i)
    # Then convert back to original x scale.
    
    # Strategy: 
    # 1. Use K-fold cross-validation to select regularization parameter (lambda) for ridge regression.
    # 2. Fit final model on all data with best lambda.
    # 3. Convert coefficients from standardized to original scale.
    
    # Build Vandermonde matrix for standardized x
    V = np.vander(x_norm, N=6, increasing=True)
    
    # K-fold CV for lambda selection
    k_folds = 5
    np.random.seed(42)
    indices = np.random.permutation(n)
    fold_size = n // k_folds
    
    # Candidate lambdas: log-spaced from 1e-8 to 1e2
    log_lams = np.linspace(np.log10(1e-8), np.log10(1e2), 20)
    lam_candidates = 10 ** log_lams
    
    best_val_mse = np.inf
    best_lam = 1e-4
    
    for i in range(k_folds):
        if time.time() - t_start > deadline - 10:
            break
        
        val_start = i * fold_size
        val_end = val_start + fold_size
        if i == k_folds - 1:
            val_end = n
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        x_train = x_norm[train_idx]
        y_train = y[train_idx]
        x_val = x_norm[val_idx]
        y_val = y[val_idx]
        
        V_train = np.vander(x_train, N=6, increasing=True)
        V_val = np.vander(x_val, N=6, increasing=True)
        
        for lam in lam_candidates:
            if time.time() - t_start > deadline - 5:
                break
            
            A = V_train.T @ V_train + lam * np.eye(6)
            b = V_train.T @ y_train
            
            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
            if not np.all(np.isfinite(coeff)):
                continue
            
            pred_val = V_val @ coeff
            mse = np.mean((pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_lam = lam
    
    # Refit on all data with best lambda
    A = V.T @ V + best_lam * np.eye(6)
    b = V.T @ y
    
    try:
        coeff_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
    
    if not np.all(np.isfinite(coeff_norm)):
        coeff_norm = np.zeros(6)
    
    # Convert from normalized x to original x scale
    # x_norm = (x - x_mean) / x_std
    # So x = x_mean + x_std * x_norm
    # We have: f(x) = sum(a_i * x_norm^i) = sum(a_i * ((x - x_mean) / x_std)^i)
    #         = sum(a_i / x_std^i * (x - x_mean)^i)
    # Expand (x - x_mean)^i = sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j)
    # So f(x) = sum_{i=0}^{5} a_i / x_std^i * sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j)
    #         = sum_{j=0}^{5} x^j * [sum_{i=j}^{5} a_i / x_std^i * C(i,j) * (-x_mean)^(i-j)]
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    for i in range(6):
        a_i = coeff_norm[i]
        if abs(a_i) < 1e-15:
            continue
        a_i_scaled = a_i / (x_std ** i)
        for j in range(i + 1):
            # C(i, j)
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
            term = binom_val * ((-x_mean) ** (i - j)) * a_i_scaled
            coeff_orig[j] += term
    
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
    
    return coeff_orig
