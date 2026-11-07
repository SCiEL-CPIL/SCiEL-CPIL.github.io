import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore")

_GLOBAL_START_TIME = None
TIME_LIMIT = 295.0

def _get_time_left():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return TIME_LIMIT - (time.time() - _GLOBAL_START_TIME)

def _matern52_kernel(X1, X2, length_scale):
    """
    Compute the Matern 5/2 kernel matrix between X1 and X2.
    X1: (N1, 1), X2: (N2, 1)
    """
    # Compute pairwise distances
    # Using broadcasting: (N1, 1) - (1, N2) -> (N1, N2)
    diff = X1 - X2.T
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Matern 5/2 formula
    # k(r) = (1 + sqrt(3)*r/l + 3*r^2/(2*l^2)) * exp(-sqrt(3)*r/l)
    # We assume signal variance sigma_f^2 = 1 for the base kernel, 
    # as we will scale y or handle sigma_f separately.
    # Actually, it's better to include sigma_f^2 in the optimization or fix it.
    # Let's fix sigma_f^2 = 1 and normalize y to have unit variance.
    # Then the kernel output is the covariance.
    
    r_over_l = dist / length_scale
    k = (1.0 + np.sqrt(3.0) * r_over_l + 1.5 * r_over_l**2) * np.exp(-np.sqrt(3.0) * r_over_l)
    
    return k

def _fit_gp(X, y, length_scale, noise=1e-6):
    """
    Fit a Gaussian Process regression model.
    X: (N, 1)
    y: (N,)
    Returns: alpha (N,) such that prediction is K(X_test, X_train) @ alpha
    """
    N = X.shape[0]
    K = _matern52_kernel(X, X, length_scale)
    K += noise * np.eye(N)
    
    try:
        # Cholesky decomposition for stability
        L = np.linalg.cholesky(K)
        # Solve L @ L.T @ alpha = y
        # L @ L.T = K
        # We can solve K @ alpha = y using cho_solve
        alpha = cho_solve((L, True), y)
        return alpha, L
    except np.linalg.LinAlgError:
        # Fallback to lstsq if Cholesky fails
        try:
            alpha = np.linalg.lstsq(K, y, rcond=None)[0]
            # Return a dummy L or None to indicate failure
            return alpha, None
        except:
            return None, None

def _predict_gp(X_test, X_train, y_train, length_scale, noise=1e-6):
    """
    Predict using a pre-trained GP (implicitly via alpha).
    For simplicity in this structure, we recompute alpha or store it.
    Here we just compute the prediction given the parameters.
    """
    N = X_train.shape[0]
    K_train = _matern52_kernel(X_train, X_train, length_scale)
    K_train += noise * np.eye(N)
    
    try:
        alpha = np.linalg.solve(K_train, y_train)
    except:
        alpha = np.linalg.lstsq(K_train, y_train, rcond=None)[0]
        
    K_test_train = _matern52_kernel(X_test, X_train, length_scale)
    return K_test_train @ alpha

def _cross_validate_gp(X, y, length_scales, n_folds=5, noise=1e-6):
    """
    Cross-validate GP over a grid of length scales.
    Returns the best length scale and its mean CV MSE.
    """
    n = X.shape[0]
    indices = np.arange(n)
    # Use a fixed seed for reproducibility in CV splits
    rng = np.random.default_rng(42)
    indices = rng.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_l = length_scales[0]
    
    # Pre-compute y variance for normalization if needed, 
    # but we are just looking at raw MSE for CV selection.
    y_var = np.var(y)
    if y_var < 1e-8:
        y_var = 1.0
    
    for l in length_scales:
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Fit GP on training fold
            alpha, _ = _fit_gp(X_tr, y_tr, l, noise)
            if alpha is None:
                fold_mses.append(np.inf)
                continue
                
            # Predict on validation fold
            K_val_tr = _matern52_kernel(X_val, X_tr, l)
            pred = K_val_tr @ alpha
            
            mse = np.mean((pred - y_val) ** 2)
            if not np.isfinite(mse):
                mse = np.inf
            fold_mses.append(mse)
            
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_l = l
            
    return best_l, best_mse

def fit(X_train, y_train):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Ensure 1D
    if d != 1:
        # Fallback for multi-dim (should not happen per task description)
        # Use simple RBF
        x = X[:, 0]
        y_mean = y.mean()
        y_std = y.std()
        if y_std < 1e-8: y_std = 1.0
        y_norm = (y - y_mean) / y_std
        
        # Simple RBF kernel method
        gamma = 1.0 / (np.var(X[:, 0]) + 1e-6)
        K = np.exp(-gamma * np.sum((X[:, None, 0] - X[None, :, 0])**2, axis=2))
        K += 1e-2 * np.eye(n)
        w = np.linalg.solve(K, y_norm)
        
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            K_t = np.exp(-gamma * np.sum((x_t[:, None] - X[:, 0][None, :])**2, axis=2))
            pred_norm = K_t @ w
            return pred_norm * y_std + y_mean
        return predict

    x = X[:, 0]
    
    # Normalize y to unit variance for GP fitting
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Define length scale grid
    # The range of x is small (e.g., [-3, 3] or [0.1, 6]).
    # Length scales should cover from very local (0.1) to global (10+)
    length_scales = np.logspace(-1, 1.5, 15) # 0.1 to ~31
    
    # Check time
    if _get_time_left() < 10.0:
        # Fallback to cubic polynomial
        x_mean = x.mean()
        x_std = x.std()
        if x_std < 1e-8: x_std = 1.0
        x_norm = (x - x_mean) / x_std
        X_poly = np.column_stack([np.ones(n), x_norm, x_norm**2, x_norm**3])
        w_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]
        
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            x_t_norm = (x_t - x_mean) / x_std
            X_t_poly = np.column_stack([np.ones(len(x_t)), x_t_norm, x_t_norm**2, x_t_norm**3])
            return X_t_poly @ w_poly
        return predict

    # Cross-validate to find best length scale
    best_l, best_cv_mse = _cross_validate_gp(x.reshape(-1, 1), y_norm, length_scales)
    
    # Check time
    if _get_time_left() < 5.0:
        # Use a default length scale if CV is too slow? 
        # We have best_l if CV finished. If not, use median of grid.
        if not np.isfinite(best_cv_mse):
            best_l = np.median(length_scales)

    # Fit final GP model on all data with best length scale
    alpha, L = _fit_gp(x.reshape(-1, 1), y_norm, best_l)
    
    # Sanity check
    if alpha is None or not np.all(np.isfinite(alpha)):
        # Fallback to polynomial if GP fails
        x_mean = x.mean()
        x_std = x.std()
        if x_std < 1e-8: x_std = 1.0
        x_norm = (x - x_mean) / x_std
        X_poly = np.column_stack([np.ones(n), x_norm, x_norm**2, x_norm**3, x_norm**4])
        w_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]
        
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            x_t_norm = (x_t - x_mean) / x_std
            X_t_poly = np.column_stack([np.ones(len(x_t)), x_t_norm, x_t_norm**2, x_t_norm**3, x_t_norm**4])
            return X_t_poly @ w_poly
        return predict

    # Store training data and parameters for prediction
    x_train = x
    y_mean_val = y_mean
    y_std_val = y_std
    best_l_val = best_l
    alpha_val = alpha

    def predict(X_test):
        X_t = np.asarray(X_test, dtype=float)
        x_t = X_t[:, 0].reshape(-1, 1)
        
        # Compute kernel between test and train
        K_test_train = _matern52_kernel(x_t, x_train.reshape(-1, 1), best_l_val)
        
        # Prediction in normalized space
        pred_norm = K_test_train @ alpha_val
        
        # Denormalize
        return pred_norm * y_std_val + y_mean_val

    return predict
