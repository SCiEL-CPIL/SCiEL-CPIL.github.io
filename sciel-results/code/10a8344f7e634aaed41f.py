import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0 # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Standardize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # We will fit degree 5 polynomial
    degree = 5
    
    # Grid of lambda values for Ridge Regression
    # Log-spaced from small to large
    lambdas = np.logspace(-8, 6, 50)
    
    # Cross-Validation to find best lambda
    # Use K-Fold CV
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42) # For reproducibility in CV split
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[:n % k_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
    
    best_lambda = 1.0
    best_cv_mse = np.inf
    
    for lam in lambdas:
        if time.time() - t_start > deadline - 10:
            break
            
        mse_sum = 0.0
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]
            
            # Fit Ridge
            V_train = np.vander(x_train, N=degree + 1, increasing=True)
            A = V_train.T @ V_train + lam * np.eye(degree + 1)
            b = V_train.T @ y_train
            
            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Predict
            V_val = np.vander(x_val, N=degree + 1, increasing=True)
            pred = V_val @ coeff
            mse = np.mean((pred - y_val) ** 2)
            mse_sum += mse
        
        avg_mse = mse_sum / k_folds
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            best_lambda = lam
            
    # Final fit on all data with best_lambda
    V_all = np.vander(x_norm, N=degree + 1, increasing=True)
    A = V_all.T @ V_all + best_lambda * np.eye(degree + 1)
    b = V_all.T @ y
    
    try:
        coeff_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
        
    if not np.all(np.isfinite(coeff_norm)):
        coeff_norm = np.nan_to_num(coeff_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Convert from normalized x to original x
    # f(x) = sum(c_i * ((x - x_mean)/x_std)^i)
    # Let u = (x - x_mean)/x_std => x = x_std * u + x_mean
    # We want d_j such that sum(d_j * x^j) = sum(c_i * u^i)
    # u^i = ((x - x_mean)/x_std)^i = (1/x_std^i) * (x - x_mean)^i
    # (x - x_mean)^i = sum_{j=0}^i binom(i, j) * x^j * (-x_mean)^(i-j)
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for i in range(degree + 1):
        if i >= 6: break # We only need up to degree 5
        c_i = coeff_norm[i]
        if abs(c_i) < 1e-15:
            continue
            
        factor = c_i / (x_std ** i)
        
        for j in range(i + 1):
            # binom(i, j)
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
                
            term = binom_val * ((-x_mean) ** (i - j)) * factor
            coeff_orig[j] += term
            
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
        
    return coeff_orig
