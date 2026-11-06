import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fits a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    
    Parameters:
    training_x: Array of slow-state values.
    training_subgrid_tendency: Array of corresponding subgrid tendency values.
    
    Returns:
    Array of 6 finite real polynomial coefficients in ASCENDING powers of x.
    """
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    if n == 0:
        return np.zeros(6)

    if n < 10:
        # Fallback for very small data: simple linear fit
        try:
            coeff = np.zeros(6)
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff[:2] = c_lin
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # Standardize x to improve conditioning
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x < 1e-12:
        std_x = 1.0
    z = (x - mean_x) / std_x

    # Build design matrix for degree 5 polynomial in z
    # Columns: 1, z, z^2, z^3, z^4, z^5
    V = np.vander(z, N=6, increasing=True)
    
    n_features = 6
    
    # Use K-fold cross-validation to select lambda
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    
    n_folds = 5
    fold_size = n // n_folds
    folds = []
    start = 0
    for i in range(n_folds):
        end = start + fold_size if i < n_folds - 1 else n
        folds.append(indices[start:end])
        start = end

    # Lambda grid
    # Estimate scale from the data
    try:
        VtV_full = V.T @ V
        trace_VtV = np.trace(VtV_full)
        if trace_VtV <= 0 or not np.isfinite(trace_VtV):
            trace_VtV = n
    except:
        trace_VtV = n
        
    base_lam = trace_VtV / n_features
    
    # Use a wider range of lambdas
    lambdas = np.logspace(-8, 6, num=30) * base_lam
    
    best_lambda = lambdas[0]
    best_cv_error = np.inf
    best_coeff_std = None
    
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
            
        fold_errors = []
        
        for i in range(n_folds):
            val_idx = folds[i]
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            try:
                A = V_train.T @ V_train + lam * np.eye(n_features)
                b = V_train.T @ y_train
                
                c_std = np.linalg.solve(A, b)
                
                if not np.all(np.isfinite(c_std)):
                    continue
                
                y_pred = V_val @ c_std
                err = np.mean((y_pred - y_val)**2)
                fold_errors.append(err)
            except np.linalg.LinAlgError:
                fold_errors.append(np.inf)
        
        if len(fold_errors) > 0:
            mean_err = np.mean(fold_errors)
            if mean_err < best_cv_error:
                best_cv_error = mean_err
                best_lambda = lam

    # Final fit with best_lambda using all data
    if time.time() < deadline - 2.0:
        try:
            A_full = V.T @ V + best_lambda * np.eye(n_features)
            b_full = V.T @ y
            c_std = np.linalg.solve(A_full, b_full)
        except np.linalg.LinAlgError:
            c_std = np.linalg.lstsq(V, y, rcond=None)[0]
    else:
        c_std = np.zeros(6)
        try:
            c_std[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            pass

    if not np.all(np.isfinite(c_std)):
        return np.zeros(6)

    # Transform coefficients from standardized z to original x
    # P(z) = sum c_i z^i, z = (x - mu) / sigma
    # We want g(x) such that g(x) = P((x-mu)/sigma)
    
    c_orig = np.zeros(6)
    for i in range(6):
        ci = c_std[i]
        if ci == 0:
            continue
        sigma_pow = (1.0 / std_x) ** i
        for j in range(i + 1):
            binom_val = 1.0
            for k in range(i - j):
                binom_val *= (i - k) / (k + 1)
            coeff_val = ci * sigma_pow * binom_val * ((-mean_x) ** (i - j))
            c_orig[j] += coeff_val

    if not np.all(np.isfinite(c_orig)):
        return np.zeros(6)

    return c_orig
