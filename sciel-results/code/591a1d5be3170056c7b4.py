"""Structural change: Robust 1D Regression with rich basis, SVD ridge, and ensemble of top CV models."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _build_features(x, basis_type, param):
    """
    Build feature matrix for a given basis type.
    basis_type: 'poly', 'sin', 'exp', 'mixed'
    param: degree or number of terms
    Returns array of shape (n, p)
    """
    x = np.asarray(x).reshape(-1, 1)
    n = x.shape[0]
    
    if basis_type == 'poly':
        degree = param
        features = np.ones((n, degree + 1))
        for i in range(1, degree + 1):
            features[:, i] = x[:, 0] ** i
        return features
    elif basis_type == 'sin':
        n_terms = param
        features = np.ones((n, 1))
        for i in range(1, n_terms + 1):
            features[:, i] = np.sin(i * x[:, 0])
        return features
    elif basis_type == 'exp':
        # Exponential basis: 1, exp(x), exp(2x), ... but can be unstable
        # Use scaled version for stability
        n_terms = param
        features = np.ones((n, 1))
        for i in range(1, n_terms + 1):
            features[:, i] = np.exp(i * 0.1 * x[:, 0])
        return features
    elif basis_type == 'mixed':
        # Rich basis: poly up to 4, sin up to 4, cos up to 4
        # Total: 1 (bias) + 4 (poly) + 4 (sin) + 4 (cos) = 13 features
        features = np.ones((n, 13))
        # Poly terms: x, x^2, x^3, x^4
        for i in range(1, 5):
            features[:, i] = x[:, 0] ** i
        # Sin terms: sin(x), sin(2x), sin(3x), sin(4x)
        for i in range(1, 5):
            features[:, 4 + i] = np.sin(i * x[:, 0])
        # Cos terms: cos(x), cos(2x), cos(3x), cos(4x)
        for i in range(1, 5):
            features[:, 8 + i] = np.cos(i * x[:, 0])
        return features
    else:
        # Fallback: linear
        return np.column_stack([np.ones(n), x[:, 0]])

def _svd_ridge_fit(X, y, alpha):
    """
    SVD-based ridge regression. More numerically stable.
    X: (n, p) feature matrix (no bias column needed if we center)
    y: (n,) target
    alpha: regularization strength
    Returns weights w of shape (p,)
    """
    n, p = X.shape
    
    # Center X and y
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    # SVD of centered X
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Regularize singular values
    s_reg = s ** 2 + alpha
    # Compute Vt^T U^T y_centered / s_reg
    Vt_Ut_y = Vt @ (U.T @ y_centered)
    w_centered = (Vt_Ut_y / s_reg).T @ Vt.T
    
    # Recover intercept
    w = np.zeros(p)
    w[1:] = w_centered  # Assume first column is bias (all ones), which is centered to 0
    w[0] = y_mean - X_mean[1:] @ w_centered
    
    return w

def _svd_ridge_predict(X, w):
    return X @ w

def _cv_score(x, y, basis_type, param, alpha, n_folds=5):
    """
    Calculate CV MSE for a given basis type, param, and alpha.
    """
    n = len(x)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    fold_mses = []
    
    for i in range(n_folds):
        if _time_left() < 2:
            return np.inf
            
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        try:
            X_tr = _build_features(x_tr, basis_type, param)
            X_val = _build_features(x_val, basis_type, param)
            
            w = _svd_ridge_fit(X_tr, y_tr, alpha)
            pred = _svd_ridge_predict(X_val, w)
            
            mse = np.mean((pred - y_val)**2)
            fold_mses.append(mse)
        except Exception:
            fold_mses.append(np.inf)
            
    if not fold_mses:
        return np.inf
    return np.mean(fold_mses)

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)
    
    # Handle constant y
    if np.std(y) < 1e-10:
        mean_y = np.mean(y)
        def predict(X):
            return np.full(X.shape[0], mean_y)
        return predict
    
    # Sort data for Spline compatibility
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Candidate models: (basis_type, param, alpha)
    # We'll test a grid of basis types, params, and alphas
    # Then select top-3 by CV score and ensemble them
    
    candidates = []
    
    # Poly: degrees 1-8
    for deg in range(1, 9):
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            candidates.append(('poly', deg, alpha))
    
    # Sin: 1-6 terms
    for n_terms in range(1, 7):
        for alpha in [0.01, 0.1, 1.0]:
            candidates.append(('sin', n_terms, alpha))
    
    # Mixed basis
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        candidates.append(('mixed', 0, alpha))
    
    # Score all candidates
    scored = []
    for basis_type, param, alpha in candidates:
        if _time_left() < 5:
            break
        mse = _cv_score(x_sorted, y_sorted, basis_type, param, alpha)
        if np.isfinite(mse):
            scored.append((mse, basis_type, param, alpha))
    
    # Sort by MSE
    scored.sort(key=lambda t: t[0])
    
    # Select top-3 for ensemble
    top_k = min(3, len(scored))
    if top_k == 0:
        # Fallback to linear
        X_poly = _build_features(x_sorted, 'poly', 1)
        w = _svd_ridge_fit(X_poly, y_sorted, 1.0)
        def predict(X):
            X_test = X[:, 0].reshape(-1, 1)
            X_test_poly = _build_features(X_test, 'poly', 1)
            return _svd_ridge_predict(X_test_poly, w)
        return predict
    
    # Fit top-k models on all data
    models = []
    for mse, basis_type, param, alpha in scored[:top_k]:
        X_all = _build_features(x_sorted, basis_type, param)
        w = _svd_ridge_fit(X_all, y_sorted, alpha)
        models.append((basis_type, param, w))
    
    def predict(X):
        X_test = X[:, 0].reshape(-1, 1)
        preds = []
        for basis_type, param, w in models:
            X_test_feat = _build_features(X_test, basis_type, param)
            pred = _svd_ridge_predict(X_test_feat, w)
            preds.append(pred)
        # Average predictions
        avg_pred = np.mean(preds, axis=0)
        return np.asarray(avg_pred, dtype=float)
    
    # Safety wrapper
    mean_y = np.mean(y)
    def safe_predict(X):
        try:
            res = predict(X)
            res = np.asarray(res, dtype=float)
            res = np.where(np.isfinite(res), res, mean_y)
            return res
        except Exception:
            return np.full(X.shape[0], mean_y)
    
    return safe_predict
# EVOLVE-BLOCK-END
