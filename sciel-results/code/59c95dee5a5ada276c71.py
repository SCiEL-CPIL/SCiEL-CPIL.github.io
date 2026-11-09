"""Symbolic regression via adaptive model selection: cubic spline, polynomial, and KRR."""
import time
import numpy as np

def fit(X_train, y_train):
    """
    Fit a regression model using adaptive model selection.
    For 1D data: cubic spline interpolation (exact fit for noiseless data).
    For higher dimensions or if spline fails: Kernel Ridge Regression with CV.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    n, d = X.shape

    def time_left():
        return deadline - time.time()

    # For 1D data with noiseless training, cubic spline interpolation is near-perfect
    if d == 1:
        x = X[:, 0]
        
        # Sort by x for spline fitting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Check for duplicate x values - if present, average y for those x
        unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        
        if len(unique_x) < n:
            # Have duplicates - average y values for each unique x
            y_sum = np.zeros(len(unique_x))
            np.add.at(y_sum, inverse_indices, y_sorted)
            y_avg = y_sum / counts
        else:
            unique_x = x_sorted
            y_avg = y_sorted
        
        # Fit cubic spline
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(unique_x, y_avg, bc_type='natural')
            
            # Validate: check that the spline doesn't produce NaNs or infinities
            test_pts = unique_x
            test_vals = cs(test_pts)
            if not np.all(np.isfinite(test_vals)):
                raise ValueError("Spline produced non-finite values")
            
            def predict_spline(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                xt = X_test[:, 0]
                # Clip to training range to avoid extrapolation issues
                xt_clipped = np.clip(xt, unique_x[0], unique_x[-1])
                pred = cs(xt_clipped)
                pred = np.where(np.isfinite(pred), pred, 0.0)
                return pred
            
            # Quick validation: check training MSE
            train_pred = predict_spline(X)
            train_mse = np.mean((train_pred - y) ** 2)
            
            # If training MSE is very small, spline is likely excellent
            y_var = np.var(y)
            if y_var > 0 and train_mse / y_var < 1e-6:
                return predict_spline
            
            # Even if not perfect, spline might still be good for smooth functions
            # But let's also try other methods and compare
            
        except Exception:
            pass  # Fall through to other methods

    # For higher dimensions or if spline wasn't great, use KRR with CV
    # Normalize inputs
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_norm = (X - x_mean) / x_std

    def rbf_kernel(X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-gamma * sq_dists)

    # Try a few gamma values with minimal CV
    best_gamma = 1.0
    best_lam = 1e-3
    best_cv_mse = np.inf

    # Coarse grid for speed
    gamma_grid = np.logspace(-2, 3, 10)
    lam_grid = np.logspace(-4, 2, 5)

    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for gamma in gamma_grid:
        if time_left() < 10.0:
            break
        K_full = rbf_kernel(X_norm, X_norm, gamma)
        
        for lam in lam_grid:
            cv_errors = []
            for i in range(n_folds):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                K_tr = K_full[np.ix_(tr_idx, tr_idx)]
                K_val = K_full[np.ix_(val_idx, tr_idx)]
                
                K_reg = K_tr + lam * np.eye(len(tr_idx))
                
                try:
                    w = np.linalg.solve(K_reg, y[tr_idx])
                    pred = K_val @ w
                    cv_errors.append(np.mean((pred - y[val_idx])**2))
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_mse = np.mean(cv_errors)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_gamma = gamma
                best_lam = lam

    # Final fit
    K_train = rbf_kernel(X_norm, X_norm, best_gamma)
    K_reg_final = K_train + best_lam * np.eye(n)
    
    try:
        w_final = np.linalg.solve(K_reg_final, y)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(K_train, y, rcond=None)[0]

    def predict_krr(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, d)
        X_test_norm = (X_test - x_mean) / x_std
        K_test = rbf_kernel(X_test_norm, X_norm, best_gamma)
        try:
            pred = K_test @ w_final
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.zeros(len(X_test))

    # If we're in 1D, compare spline vs KRR on training data and pick better
    if d == 1:
        try:
            from scipy.interpolate import CubicSpline
            sort_idx = np.argsort(X[:, 0])
            x_sorted = X[sort_idx, 0]
            y_sorted = y[sort_idx]
            unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
            if len(unique_x) < n:
                y_sum = np.zeros(len(unique_x))
                np.add.at(y_sum, inverse_indices, y_sorted)
                y_avg = y_sum / counts
            else:
                unique_x = x_sorted
                y_avg = y_sorted
            
            cs = CubicSpline(unique_x, y_avg, bc_type='natural')
            
            def predict_spline2(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                xt = X_test[:, 0]
                xt_clipped = np.clip(xt, unique_x[0], unique_x[-1])
                pred = cs(xt_clipped)
                pred = np.where(np.isfinite(pred), pred, 0.0)
                return pred
            
            # Compare on training data
            spline_train_pred = predict_spline2(X)
            krr_train_pred = predict_krr(X)
            
            y_var = np.var(y)
            if y_var > 0:
                spline_nmse = np.mean((spline_train_pred - y)**2) / y_var
                krr_nmse = np.mean((krr_train_pred - y)**2) / y_var
                
                if spline_nmse < krr_nmse:
                    return predict_spline2
        
        except Exception:
            pass

    return predict_krr
