"""Symbolic regression via global RBF network with Gaussian kernels."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Explicit deadline guard: total wall-clock budget
_TOTAL_DEADLINE = 330.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _rbf_predict(X_test, centers, weights, sigma, bias):
    """Evaluate RBF network at X_test points."""
    Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
    # Compute distances to all centers
    diffs = Xt[:, None] - centers[None, :]
    sq_dists = diffs * diffs
    # Gaussian kernel: exp(-sq_dists / (2 * sigma^2))
    kernels = np.exp(-sq_dists / (2.0 * sigma * sigma))
    pred = kernels @ weights + bias
    return pred


def fit(X_train, y_train):
    """
    Fit a regression model to 1D noiseless data using a global RBF network.
    
    Strategy:
      1. Use training points as RBF centers.
      2. Determine optimal sigma via cross-validation over a range of values.
      3. Solve for weights via regularized least squares.
      4. Fallback to polynomial or linear interpolation if RBF fails.
    """
    X = np.asarray(X_train, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(X)
    
    if n < 4:
        mean_y = np.mean(y)
        return lambda X_test: np.full(len(np.asarray(X_test).reshape(-1)), mean_y)
    
    # Sort by X and remove duplicates
    sort_idx = np.argsort(X)
    Xs = X[sort_idx]
    ys = y[sort_idx]
    
    unique_mask = np.ones(len(Xs), dtype=bool)
    if len(Xs) > 1:
        unique_mask[1:] = Xs[1:] != Xs[:-1]
    Xs_u = Xs[unique_mask]
    ys_u = ys[unique_mask]
    
    x_min, x_max = Xs_u[0], Xs_u[-1]
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-12
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    
    # Use a subset of centers if n is large to keep computation manageable
    # For 200 points, using all is fine, but let's cap at 150 for safety
    max_centers = min(n, 150)
    if n > max_centers:
        # Evenly spaced subset
        indices = np.linspace(0, n - 1, max_centers, dtype=int)
        centers = Xs_u[indices]
        ys_sub = ys_u[indices]
    else:
        centers = Xs_u
        ys_sub = ys_u
    
    n_centers = len(centers)
    
    # Try RBF with different sigma values
    best_predict = None
    best_cv_score = np.inf
    
    # Candidate sigmas: scale relative to x_range
    sigma_candidates = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]) * x_range
    
    # Cross-validation: split into 5 folds
    n_folds = 5
    fold_size = n // n_folds
    indices_all = np.arange(n)
    
    for sigma in sigma_candidates:
        if _elapsed() > _TOTAL_DEADLINE - 30:
            break
            
        cv_errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices_all[val_start:val_end]
            train_idx = np.concatenate([indices_all[:val_start], indices_all[val_end:]])
            
            X_train_cv = Xs_u[train_idx]
            y_train_cv = ys_u[train_idx]
            X_val_cv = Xs_u[val_idx]
            y_val_cv = ys_u[val_idx]
            
            # Use subset of training points as centers
            n_cv_centers = min(len(X_train_cv), 100)
            if len(X_train_cv) > n_cv_centers:
                cv_center_idx = np.linspace(0, len(X_train_cv) - 1, n_cv_centers, dtype=int)
                cv_centers = X_train_cv[cv_center_idx]
                cv_y = y_train_cv[cv_center_idx]
            else:
                cv_centers = X_train_cv
                cv_y = y_train_cv
            
            # Build kernel matrix
            diffs = cv_centers[:, None] - cv_centers[None, :]
            sq_dists = diffs * diffs
            K = np.exp(-sq_dists / (2.0 * sigma * sigma))
            
            # Regularized least squares: (K + lambda*I) w = y
            # Choose lambda based on condition number
            reg = 1e-6 * np.trace(K) / len(K)
            try:
                w = np.linalg.solve(K + reg * np.eye(len(cv_centers)), cv_y)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K + reg * np.eye(len(cv_centers)), cv_y, rcond=None)[0]
            
            # Evaluate on validation set
            diffs_val = X_val_cv[:, None] - cv_centers[None, :]
            K_val = np.exp(-(diffs_val * diffs_val) / (2.0 * sigma * sigma))
            pred_val = K_val @ w
            
            mse = np.mean((pred_val - y_val_cv) ** 2)
            cv_errors.append(mse)
        
        avg_cv_mse = np.mean(cv_errors)
        
        if avg_cv_mse < best_cv_score:
            best_cv_score = avg_cv_mse
            best_sigma = sigma
    
    # Now fit final RBF with best sigma
    if best_predict is None and 'best_sigma' in locals():
        sigma = best_sigma
        diffs = centers[:, None] - centers[None, :]
        sq_dists = diffs * diffs
        K = np.exp(-sq_dists / (2.0 * sigma * sigma))
        
        # Regularized least squares
        reg = 1e-6 * np.trace(K) / len(K)
        try:
            weights = np.linalg.solve(K + reg * np.eye(n_centers), ys_sub)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(K + reg * np.eye(n_centers), ys_sub, rcond=None)[0]
        
        bias = 0.0
        
        # Verify on training data
        pred_train = _rbf_predict(Xs_u, centers, weights, sigma, bias)
        train_mse = np.mean((pred_train - ys_u) ** 2)
        
        if np.all(np.isfinite(pred_train)) and train_mse < 1e-4 * (y_std ** 2):
            def predict_rbf(X_test):
                Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
                pred = _rbf_predict(Xt, centers, weights, sigma, bias)
                pred = np.where(np.isfinite(pred), pred, y_mean)
                return pred
            
            return predict_rbf
    
    # Fallback: cubic spline
    try:
        from scipy.interpolate import CubicSpline
        if len(Xs_u) >= 4:
            cs = CubicSpline(Xs_u, ys_u, bc_type='not-a-knot')
            test_pred = cs(Xs_u)
            if np.all(np.isfinite(test_pred)):
                train_mse = np.mean((test_pred - ys_u) ** 2)
                if train_mse < 1e-6 * (y_std ** 2):
                    def predict_spline(X_test):
                        Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
                        Xt_clipped = np.clip(Xt, x_min, x_max)
                        pred = cs(Xt_clipped)
                        pred = np.where(np.isfinite(pred), pred, y_mean)
                        return pred
                    return predict_spline
    except Exception:
        pass
    
    # Fallback: polynomial fit
    try:
        x_mid = (x_min + x_max) / 2
        Xs_scaled = (Xs_u - x_mid) / (x_range / 2)
        deg = min(6, len(Xs_u) - 1)
        coeffs = np.polyfit(Xs_scaled, ys_u, deg)
        
        def predict_poly(X_test):
            Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
            Xt_scaled = (Xt - x_mid) / (x_range / 2)
            pred = np.polyval(coeffs, Xt_scaled)
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        return predict_poly
    except Exception:
        pass
    
    # Final fallback: linear interpolation
    try:
        from scipy.interpolate import interp1d
        f_interp = interp1d(Xs_u, ys_u, kind='linear', fill_value='extrapolate', bounds_error=False)
        
        def predict_linear(X_test):
            Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
            pred = f_interp(Xt)
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        return predict_linear
    except Exception:
        pass
    
    # Last resort: mean
    def predict_mean(X_test):
        Xt = np.asarray(X_test, dtype=np.float64).reshape(-1)
        return np.full(len(Xt), y_mean)
    
    return predict_mean
# EVOLVE-BLOCK-END
