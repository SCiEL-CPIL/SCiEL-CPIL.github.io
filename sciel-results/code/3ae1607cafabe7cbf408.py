"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    # We'll fit in scaled space: z = (x - x_mean) / x_std
    # Polynomial in z: p(z) = sum c_i_scaled * z^i
    # Then convert back to polynomial in x

    z = (x - x_mean) / x_std

    def check_time():
        return time.time() < deadline - 5

    # Try different degrees and regularization approaches
    best_coeff = None
    best_val = np.inf

    # Approach 1: Direct least squares with ridge regularization for various degrees
    for degree in [1, 2, 3, 4, 5]:
        if not check_time():
            break
        # Build Vandermonde matrix in scaled space
        V = np.vander(z, N=degree + 1, increasing=True)
        
        # Try a few regularization strengths
        for lam in [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
            if not check_time():
                break
            try:
                A = V.T @ V
                b = V.T @ y
                # Add regularization
                reg = np.eye(degree + 1) * lam
                reg[0, 0] = 0  # Don't regularize intercept as much
                A_reg = A + reg
                c_scaled = np.linalg.solve(A_reg, b)
                
                # Convert to original x space: x = x_mean + x_std * z
                # p(x) = p((x - x_mean)/x_std)
                # Expand: sum c_scaled_i * ((x - x_mean)/x_std)^i
                # = sum c_scaled_i * x_std^(-i) * (x - x_mean)^i
                # (x - x_mean)^i = sum_{k=0}^i binom(i,k) * x_mean^(i-k) * x^k
                coeff_x = np.zeros(degree + 1)
                for i in range(degree + 1):
                    if c_scaled[i] == 0:
                        continue
                    ci = c_scaled[i] / (x_std ** i)
                    # (x - x_mean)^i = sum_{k=0}^i C(i,k) * x^k * (-x_mean)^(i-k)
                    for k in range(i + 1):
                        from math import comb
                        coeff_x[k] += ci * comb(i, k) * ((-x_mean) ** (i - k))
                
                # Compute training error (normalized)
                pred = np.zeros_like(y)
                for i in range(degree + 1):
                    pred += coeff_x[i] * (x ** i)
                mse = np.mean((pred - y) ** 2)
                var_y = np.var(y)
                if var_y > 1e-12:
                    norm_err = mse / var_y
                else:
                    norm_err = mse
                
                if norm_err < best_val:
                    best_val = norm_err
                    best_coeff = np.zeros(6)
                    best_coeff[:degree + 1] = coeff_x[:degree + 1]
            except Exception:
                continue

    # Approach 2: Weighted least squares with adaptive weights (soft approach)
    if check_time() and best_coeff is not None:
        # Use the best degree from approach 1
        # Re-fit with more careful weighting
        for degree in [3, 4, 5]:
            if not check_time():
                break
            try:
                V = np.vander(z, N=degree + 1, increasing=True)
                # Iteratively reweighted least squares
                c_scaled = np.zeros(degree + 1)
                for iteration in range(20):
                    if not check_time():
                        break
                    pred = V @ c_scaled
                    residual = y - pred
                    # Soft weighting: downweight large residuals
                    scale = np.median(np.abs(residual)) + 1e-8
                    w = np.exp(-np.abs(residual) / (2 * scale))
                    W = np.diag(w)
                    A = V.T @ W @ V
                    b = V.T @ W @ y
                    reg = np.eye(degree + 1) * 1e-6
                    reg[0, 0] = 0
                    c_new = np.linalg.solve(A + reg, b)
                    if np.max(np.abs(c_new - c_scaled)) < 1e-10:
                        c_scaled = c_new
                        break
                    c_scaled = c_new
                
                # Convert to x space
                coeff_x = np.zeros(degree + 1)
                for i in range(degree + 1):
                    if c_scaled[i] == 0:
                        continue
                    ci = c_scaled[i] / (x_std ** i)
                    from math import comb
                    for k in range(i + 1):
                        coeff_x[k] += ci * comb(i, k) * ((-x_mean) ** (i - k))
                
                pred = np.zeros_like(y)
                for i in range(degree + 1):
                    pred += coeff_x[i] * (x ** i)
                mse = np.mean((pred - y) ** 2)
                var_y = np.var(y)
                if var_y > 1e-12:
                    norm_err = mse / var_y
                else:
                    norm_err = mse
                
                if norm_err < best_val:
                    best_val = norm_err
                    best_coeff = np.zeros(6)
                    best_coeff[:degree + 1] = coeff_x[:degree + 1]
            except Exception:
                continue

    # Approach 3: Cross-validation on training data to pick best polynomial
    if check_time() and best_coeff is not None:
        # Simple k-fold CV to validate
        n_folds = 5
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
        folds[-1] = indices[(n_folds - 1) * fold_size:]  # Last fold gets remainder

        cv_errors = []
        for degree in [1, 2, 3, 4, 5]:
            if not check_time():
                break
            fold_mses = []
            for fold_idx in range(n_folds):
                val_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
                
                x_train = x[train_idx]
                y_train = y[train_idx]
                z_train = (x_train - x_mean) / x_std
                
                V_train = np.vander(z_train, N=degree + 1, increasing=True)
                A = V_train.T @ V_train
                b = V_train.T @ y_train
                reg = np.eye(degree + 1) * 1e-4
                reg[0, 0] = 0
                try:
                    c_scaled = np.linalg.solve(A + reg, b)
                except Exception:
                    c_scaled = np.linalg.lstsq(A + reg, b, rcond=None)[0]
                
                # Convert to x space
                coeff_x = np.zeros(degree + 1)
                for i in range(degree + 1):
                    if c_scaled[i] == 0:
                        continue
                    ci = c_scaled[i] / (x_std ** i)
                    from math import comb
                    for k in range(i + 1):
                        coeff_x[k] += ci * comb(i, k) * ((-x_mean) ** (i - k))
                
                # Evaluate on validation fold
                pred_val = np.zeros(len(val_idx))
                x_val = x[val_idx]
                for i in range(degree + 1):
                    pred_val += coeff_x[i] * (x_val ** i)
                mse_fold = np.mean((pred_val - y[val_idx]) ** 2)
                fold_mses.append(mse_fold)
            
            cv_err = np.mean(fold_mses)
            cv_errors.append((cv_err, degree))
        
        if cv_errors:
            cv_errors.sort()
            best_cv_degree = cv_errors[0][1]
            
            # Final fit with best CV degree
            if check_time():
                try:
                    V = np.vander(z, N=best_cv_degree + 1, increasing=True)
                    A = V.T @ V
                    b = V.T @ y
                    reg = np.eye(best_cv_degree + 1) * 1e-4
                    reg[0, 0] = 0
                    c_scaled = np.linalg.solve(A + reg, b)
                    
                    coeff_x = np.zeros(best_cv_degree + 1)
                    for i in range(best_cv_degree + 1):
                        if c_scaled[i] == 0:
                            continue
                        ci = c_scaled[i] / (x_std ** i)
                        from math import comb
                        for k in range(i + 1):
                            coeff_x[k] += ci * comb(i, k) * ((-x_mean) ** (i - k))
                    
                    # Use this if it looks reasonable
                    if np.all(np.isfinite(coeff_x)):
                        best_coeff = np.zeros(6)
                        best_coeff[:best_cv_degree + 1] = coeff_x[:best_cv_degree + 1]
                except Exception:
                    pass

    # Ensure all coefficients are finite
    if best_coeff is None or not np.all(np.isfinite(best_coeff)):
        # Fallback to simple polyfit
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            if not np.all(np.isfinite(best_coeff)):
                best_coeff = np.zeros(6)
        except Exception:
            best_coeff = np.zeros(6)

    # Final safety check
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff.astype(np.float64)
# EVOLVE-BLOCK-END
