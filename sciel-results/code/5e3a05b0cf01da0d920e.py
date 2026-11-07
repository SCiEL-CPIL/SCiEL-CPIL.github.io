"""Structural change: LOESS with adaptive bandwidth + Chebyshev/Fourier ridge fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _loess_predict(x_train, y_train, x_test, bandwidth, degree=2):
    """
    Local polynomial regression (LOESS) for 1D data.
    bandwidth: fraction of data points used in local window (e.g., 0.3 means 30% of points)
    degree: polynomial degree for local fit
    """
    n = len(x_train)
    k = max(degree + 1, int(bandwidth * n))  # number of neighbors
    
    x_sorted_idx = np.argsort(x_train)
    x_sorted = x_train[x_sorted_idx]
    y_sorted = y_train[x_sorted_idx]
    
    preds = np.zeros(len(x_test))
    
    for i, xt in enumerate(x_test):
        if _time_left() < 2:
            # Fallback to nearest neighbor if running out of time
            dists = np.abs(x_sorted - xt)
            nearest = np.argmin(dists)
            preds[i] = y_sorted[nearest]
            continue
        
        # Find k nearest neighbors
        dists = np.abs(x_sorted - xt)
        # Use argpartition for efficiency
        if k < n:
            part_idx = np.argpartition(dists, k)[:k]
        else:
            part_idx = np.arange(n)
        
        # Get the actual indices of the k nearest
        nearest_dists = dists[part_idx]
        nearest_indices = part_idx
        
        # Weights: tricube kernel
        max_dist = np.max(nearest_dists) if len(nearest_dists) > 0 else 1.0
        if max_dist < 1e-10:
            weights = np.ones(k)
        else:
            u = nearest_dists / max_dist
            weights = (1 - u**3)**3
        
        # Handle case where all weights are zero (shouldn't happen with tricube)
        if np.sum(weights) < 1e-10:
            weights = np.ones(k)
        
        # Local weighted polynomial fit
        x_local = x_sorted[nearest_indices]
        y_local = y_sorted[nearest_indices]
        
        # Build Vandermonde matrix for local polynomial
        # Use centered x for numerical stability
        x_center = np.mean(x_local)
        x_scaled = (x_local - x_center) / (np.std(x_local) + 1e-10)
        
        V = np.vander(x_scaled, N=degree+1, increasing=True)
        W = np.diag(weights)
        
        try:
            # Weighted least squares: (V^T W V) w = V^T W y
            VtW = V.T * weights  # (degree+1, k)
            A = VtW @ V  # (degree+1, degree+1)
            b = VtW @ y_local  # (degree+1,)
            
            # Add small regularization for stability
            A += 1e-8 * np.eye(degree+1)
            w = np.linalg.solve(A, b)
            
            # Predict
            xt_scaled = (xt - x_center) / (np.std(x_local) + 1e-10)
            v_test = np.vander([xt_scaled], N=degree+1, increasing=True)[0]
            preds[i] = v_test @ w
            
        except Exception:
            # Fallback to weighted mean
            preds[i] = np.sum(weights * y_local) / np.sum(weights)
    
    return preds

def _loocv_loess(x, y, bandwidth, degree=2):
    """
    Leave-one-out CV for LOESS.
    """
    n = len(x)
    k = max(degree + 1, int(bandwidth * n))
    
    # For LOOCV with LOESS, we can use the hat matrix approach or just do it directly
    # Direct approach is O(n^2 * degree) which is fine for n=200
    residuals = np.zeros(n)
    
    for i in range(n):
        if _time_left() < 2:
            return np.inf
        
        x_train_i = np.delete(x, i)
        y_train_i = np.delete(y, i)
        x_test_i = np.array([x[i]])
        
        pred = _loess_predict(x_train_i, y_train_i, x_test_i, bandwidth, degree)
        residuals[i] = y[i] - pred[0]
    
    return np.mean(residuals**2)

def _chebyshev_fourier_features(x, n_cheb, n_fourier, x_min, x_max):
    """
    Generate Chebyshev and Fourier basis features.
    x: 1D array
    n_cheb: number of Chebyshev terms
    n_fourier: number of Fourier terms
    x_min, x_max: range for normalization
    """
    n = len(x)
    # Normalize x to [-1, 1] for Chebyshev
    x_norm = (2 * x - (x_max + x_min)) / (x_max - x_min + 1e-10)
    
    features = np.ones((n, 1))  # bias term
    
    # Chebyshev polynomials T_k(x)
    if n_cheb > 0:
        T0 = np.ones(n)
        T1 = x_norm.copy()
        features = np.column_stack([features, T0])
        if n_cheb > 1:
            features = np.column_stack([features, T1])
            for k in range(2, n_cheb + 1):
                Tk = 2 * x_norm * T1 - T0
                features = np.column_stack([features, Tk])
                T0, T1 = T1, Tk
    
    # Fourier terms: sin(k*pi*x/L) and cos(k*pi*x/L)
    # Use the actual range for period
    L = x_max - x_min + 1e-10
    for k in range(1, n_fourier + 1):
        features = np.column_stack([features, np.sin(2 * np.pi * k * x / L)])
        features = np.column_stack([features, np.cos(2 * np.pi * k * x / L)])
    
    return features

def _ridge_fit(x_features, y, alpha=1.0):
    """
    Fit Ridge regression.
    """
    n, p = x_features.shape
    
    A = x_features.T @ x_features
    b = x_features.T @ y
    
    reg = alpha * np.eye(p)
    reg[0, 0] = 0.0  # Don't regularize bias
    
    A_reg = A + reg
    
    try:
        w = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(x_features, y, rcond=None)[0]
    
    return w

def _ridge_predict(x_features, w):
    return x_features @ w

def _loocv_ridge(x, y, n_cheb, n_fourier, x_min, x_max, alpha=1.0):
    """
    LOOCV for Ridge with Chebyshev+Fourier basis.
    Uses efficient LOOCV formula for ridge: 
    LOO prediction = y_i - (X_i @ w) / (1 + X_i @ (X^T X + alpha I)^{-1} X_i^T)
    """
    n = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    
    X = _chebyshev_fourier_features(x, n_cheb, n_fourier, x_min, x_max)
    
    A = X.T @ X
    b = X.T @ y
    
    reg = alpha * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    
    A_reg = A + reg
    
    try:
        # Compute inverse of A_reg for LOOCV
        A_inv = np.linalg.inv(A_reg)
        w = A_inv @ b
        
        # LOOCV: e_i = y_i - x_i^T w
        # h_i = x_i^T A_inv x_i
        # LOO pred = y_i - e_i / (1 + h_i)
        preds = X @ w
        residuals = y - preds
        
        # Compute h_i = x_i^T A_inv x_i
        XAinv = X @ A_inv
        h = np.sum(X * XAinv, axis=1)
        
        loo_preds = y - residuals / (1 + h)
        mse = np.mean((loo_preds - y)**2)
        
    except Exception:
        # Fallback to direct LOOCV
        mse = np.inf
        for i in range(n):
            if _time_left() < 2:
                return np.inf
            X_train_i = np.delete(X, i, axis=0)
            y_train_i = np.delete(y, i)
            w_i = _ridge_fit(X_train_i, y_train_i, alpha)
            pred_i = _ridge_predict(X[i:i+1], w_i)
            if np.isinf(mse):
                mse = (y[i] - pred_i[0])**2
            else:
                mse = mse + (y[i] - pred_i[0])**2
        mse = mse / n
    
    return mse

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)
    
    # Handle constant y
    y_std = np.std(y)
    if y_std < 1e-10:
        def predict(X):
            return np.full(X.shape[0], np.mean(y))
        return predict
    
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Candidate models
    # 1. LOESS with different bandwidths
    # 2. Chebyshev+Fourier Ridge with different dimensions
    
    best_mse = np.inf
    best_model = None
    best_params = None
    
    # Try LOESS with different bandwidths
    bandwidths = [0.2, 0.3, 0.4, 0.5, 0.6]
    for bw in bandwidths:
        if _time_left() < 10:
            break
        mse = _loocv_loess(x, y, bw, degree=2)
        if mse < best_mse:
            best_mse = mse
            best_model = 'loess'
            best_params = {'bandwidth': bw, 'degree': 2}
    
    # Try Chebyshev+Fourier Ridge
    # Grid over n_cheb and n_fourier
    cheb_grid = [3, 5, 8, 12]
    fourier_grid = [0, 2, 4, 6]
    
    for nc in cheb_grid:
        for nf in fourier_grid:
            if _time_left() < 10:
                break
            mse = _loocv_ridge(x, y, nc, nf, x_min, x_max, alpha=1.0)
            if mse < best_mse:
                best_mse = mse
                best_model = 'ridge'
                best_params = {'n_cheb': nc, 'n_fourier': nf, 'alpha': 1.0}
    
    # If no model found, fallback to linear
    if best_model is None:
        best_model = 'ridge'
        best_params = {'n_cheb': 1, 'n_fourier': 0, 'alpha': 1.0}
    
    # Fit final model on all data
    if best_model == 'loess':
        bw = best_params['bandwidth']
        degree = best_params['degree']
        
        def predict(X):
            X_test = X[:, 0]
            return _loess_predict(x, y, X_test, bw, degree)
    
    else:  # ridge
        nc = best_params['n_cheb']
        nf = best_params['n_fourier']
        alpha = best_params['alpha']
        
        X_features = _chebyshev_fourier_features(x, nc, nf, x_min, x_max)
        w = _ridge_fit(X_features, y, alpha)
        
        def predict(X):
            X_test = X[:, 0]
            X_test_features = _chebyshev_fourier_features(X_test, nc, nf, x_min, x_max)
            return _ridge_predict(X_test_features, w)
    
    # Safety wrapper
    def safe_predict(X):
        try:
            res = predict(X)
            res = np.asarray(res, dtype=float)
            res = np.where(np.isfinite(res), res, np.mean(y))
            return res
        except Exception:
            return np.full(X.shape[0], np.mean(y))
    
    return safe_predict
# EVOLVE-BLOCK-END
