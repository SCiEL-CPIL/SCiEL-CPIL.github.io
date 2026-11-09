"""Symbolic regression with Kernel Ridge Regression and adaptive kernel selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def _rbf_kernel(X1, X2, lengthscale):
    """Compute RBF kernel between two sets of points."""
    # Normalize
    X1_mean = X1.mean(axis=0)
    X1_std = X1.std(axis=0) + 1e-10
    X2_mean = X2.mean(axis=0)
    X2_std = X2.std(axis=0) + 1e-10
    
    X1n = (X1 - X1_mean) / X1_std
    X2n = (X2 - X2_mean) / X2_std
    
    diff = X1n[:, np.newaxis, :] - X2n[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-0.5 * dist_sq / (lengthscale ** 2))

def _poly_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """Compute polynomial kernel."""
    X1_mean = X1.mean(axis=0)
    X1_std = X1.std(axis=0) + 1e-10
    X2_mean = X2.mean(axis=0)
    X2_std = X2.std(axis=0) + 1e-10
    
    X1n = (X1 - X1_mean) / X1_std
    X2n = (X2 - X2_mean) / X2_std
    
    return (gamma * (X1n @ X2n.T) + coef0) ** degree

def _sines_kernel(X1, X2, max_freq=10):
    """Compute Sum-of-Sines kernel: sum of cosines at different frequencies."""
    X1_mean = X1.mean(axis=0)
    X1_std = X1.std(axis=0) + 1e-10
    X2_mean = X2.mean(axis=0)
    X2_std = X2.std(axis=0) + 1e-10
    
    X1n = (X1 - X1_mean) / X1_std
    X2n = (X2 - X2_mean) / X2_std
    
    # Sum of cosines: K(x, x') = sum_{k=1}^{K} cos(2*pi*k*(x-x'))
    # This is equivalent to a Fourier feature map
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for k in range(1, max_freq + 1):
        # For each dimension, compute cos(2*pi*k*(x_i - x'_i))
        # K = sum over dimensions of product of cos terms
        # Actually, the sum-of-sines kernel is:
        # K(x, x') = prod_d [sum_k cos(2*pi*k*(x_d - x'_d))]
        # But that's expensive. Let's use a simpler version:
        # K(x, x') = sum_k prod_d cos(2*pi*k*(x_d - x'_d))
        # For 1D: K = sum_k cos(2*pi*k*(x - x'))
        # For multi-D: K = sum_k prod_d cos(2*pi*k*(x_d - x'_d))
        
        # Compute phase differences
        diff = X1n[:, np.newaxis, :] - X2n[np.newaxis, :, :]
        # For each frequency k, compute product of cosines over dimensions
        # Then sum over k
        prod_cos = np.ones((X1.shape[0], X2.shape[0]))
        for d in range(diff.shape[2]):
            prod_cos *= np.cos(2 * np.pi * k * diff[:, :, d])
        K += prod_cos
    return K

def _linear_kernel(X1, X2):
    """Compute linear kernel."""
    X1_mean = X1.mean(axis=0)
    X1_std = X1.std(axis=0) + 1e-10
    X2_mean = X2.mean(axis=0)
    X2_std = X2.std(axis=0) + 1e-10
    
    X1n = (X1 - X1_mean) / X1_std
    X2n = (X2 - X2_mean) / X2_std
    
    return X1n @ X2n.T

def _fit_krr(X_train, y_train, X_test, kernel_fn, alpha):
    """Fit Kernel Ridge Regression and predict on X_test."""
    n = X_train.shape[0]
    K_train = kernel_fn(X_train, X_train)
    K_train += alpha * np.eye(n)
    
    try:
        # Solve (K + alpha*I) w = y
        w = np.linalg.solve(K_train, y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_train, y_train, rcond=None)[0]
    
    K_test = kernel_fn(X_test, X_train)
    pred = K_test @ w
    return pred

def _cv_krr(X, y, kernel_fn, alpha, n_folds=5):
    """Cross-validate KRR with given kernel and alpha."""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    mse_sum = 0
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        try:
            pred = _fit_krr(X_tr, y_tr, X_val, kernel_fn, alpha)
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse_sum += np.mean((pred - y_val) ** 2)
        except Exception:
            return np.inf
    
    return mse_sum / n_folds

def _cv_krr_multi_alpha(X, y, kernel_fn, alpha_grid, n_folds=5):
    """Cross-validate KRR with multiple alphas."""
    best_score = np.inf
    best_alpha = alpha_grid[0]
    
    for alpha in alpha_grid:
        score = _cv_krr(X, y, kernel_fn, alpha, n_folds)
        if score < best_score:
            best_score = score
            best_alpha = alpha
    
    return best_score, best_alpha

def _fit_ridge(X, y, lam=1e-6):
    """Fit ridge regression, return weights and intercept."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    b = Xb.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _predict_ridge(X, w):
    """Predict using ridge model."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    return Xb @ w

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    # Degree 2
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # Degree 3 (only if degree >= 3)
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _rbf_features(X, centers, gamma):
    """Generate RBF features."""
    n = X.shape[0]
    d = X.shape[1]
    center_mean = centers.mean(axis=0)
    center_std = centers.std(axis=0) + 1e-10
    X_norm = (X - center_mean) / center_std
    C_norm = (centers - center_mean) / center_std
    
    diff = X_norm[:, np.newaxis, :] - C_norm[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * dist_sq)

def _fourier_features(X, n_freqs=5):
    """Generate Fourier features."""
    n, d = X.shape
    cols = [np.ones(n)]
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    for freq in range(1, n_freqs + 1):
        for j in range(d):
            cols.append(np.sin(2 * np.pi * freq * X_norm[:, j]))
            cols.append(np.cos(2 * np.pi * freq * X_norm[:, j]))
    return np.column_stack(cols)

def _hybrid_features(X, n_freqs=5):
    """Hybrid: linear + quadratic + cubic pure + sin/cos at multiple frequencies."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    for i in range(d):
        cols.append(X[:, i] ** 3)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    for freq in range(1, n_freqs + 1):
        for j in range(d):
            cols.append(np.sin(2 * np.pi * freq * X_norm[:, j]))
            cols.append(np.cos(2 * np.pi * freq * X_norm[:, j]))
    return np.column_stack(cols)

def _cv_score_multi_lambda(X, y, feature_fn, n_folds=5, lam_grid=None):
    """Cross-validate a feature function with lambda sweep."""
    if lam_grid is None:
        lam_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    best_score = np.inf
    best_lam = lam_grid[0]
    
    for lam in lam_grid:
        mse_sum = 0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            try:
                F_train = feature_fn(X_train)
                F_val = feature_fn(X_val)
                w = _fit_ridge(F_train, y_train, lam)
                pred = _predict_ridge(F_val, w)
                if not np.all(np.isfinite(pred)):
                    mse_sum = np.inf
                    break
                mse_sum += np.mean((pred - y_val) ** 2)
            except Exception:
                mse_sum = np.inf
                break
        if mse_sum < best_score:
            best_score = mse_sum / n_folds
            best_lam = lam
    
    return best_score, best_lam

def fit(X_train, y_train):
    """Fit a regression model using KRR with adaptive kernel selection and feature-based fallbacks."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Alpha grid for KRR
    alpha_grid = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    # Candidate kernels
    kernel_candidates = []
    
    # RBF with different lengthscales
    for ls in [0.5, 1.0, 2.0, 5.0]:
        kernel_candidates.append((f'rbf_ls{ls}', lambda X1, X2, l=ls: _rbf_kernel(X1, X2, l)))
    
    # Polynomial kernels
    for deg in [2, 3]:
        for gamma in [0.5, 1.0, 2.0]:
            kernel_candidates.append((f'poly{deg}_g{gamma}', lambda X1, X2, d=deg, g=gamma: _poly_kernel(X1, X2, d, g)))
    
    # Sum-of-Sines kernels
    for max_f in [5, 10, 15]:
        kernel_candidates.append((f'sines_f{max_f}', lambda X1, X2, mf=max_f: _sines_kernel(X1, X2, mf)))
    
    # Linear kernel
    kernel_candidates.append(('linear', lambda X1, X2: _linear_kernel(X1, X2)))
    
    # Evaluate KRR candidates
    best_score = np.inf
    best_kernel_name = None
    best_kernel_fn = None
    best_alpha = 1e-3
    
    for name, kfn in kernel_candidates:
        if time.time() > deadline - 60:
            break
        try:
            score, alpha = _cv_krr_multi_alpha(X, y, kfn, alpha_grid, n_folds=5)
            if score < best_score:
                best_score = score
                best_kernel_name = name
                best_kernel_fn = kfn
                best_alpha = alpha
        except Exception:
            continue
    
    # Also evaluate feature-based ridge models
    feature_candidates = []
    
    # Polynomial degree 2
    feature_candidates.append(('poly2', lambda Xn: _poly_features(Xn, 2)))
    
    # Polynomial degree 3
    if d <= 3:
        feature_candidates.append(('poly3', lambda Xn: _poly_features(Xn, 3)))
    
    # RBF features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    rbf_centers = X_norm[::max(1, n // 20)]
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        feature_candidates.append((f'rbf_feat_g{gamma}', lambda Xn, g=gamma, c=rbf_centers: _rbf_features(Xn, c, g)))
    
    # Fourier features
    feature_candidates.append(('fourier5', lambda Xn: _fourier_features(Xn, 5)))
    feature_candidates.append(('fourier10', lambda Xn: _fourier_features(Xn, 10)))
    
    # Hybrid features
    feature_candidates.append(('hybrid5', lambda Xn: _hybrid_features(Xn, 5)))
    
    lam_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    for name, fn in feature_candidates:
        if time.time() > deadline - 30:
            break
        try:
            score, lam = _cv_score_multi_lambda(X_norm, y, fn, n_folds=5, lam_grid=lam_grid)
            if score < best_score:
                best_score = score
                best_kernel_name = name
                best_kernel_fn = None
                best_alpha = lam
                # Store feature fn for later use
                best_feature_fn = fn
        except Exception:
            continue
    
    # Determine which model to use
    use_krr = best_kernel_fn is not None
    use_feature = not use_krr
    
    # If no valid model found, fall back to simple linear
    if use_krr:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            pred = _fit_krr(X, y, X_test, best_kernel_fn, best_alpha)
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
    elif use_feature:
        F = best_feature_fn(X_norm)
        w = _fit_ridge(F, y, lam=best_alpha)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            X_test_norm = (X_test - X_mean) / X_std
            F_test = best_feature_fn(X_test_norm)
            pred = _predict_ridge(F_test, w)
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
    else:
        # Fallback: simple linear ridge
        Xb = np.hstack([X, np.ones((n, 1))])
        w = _fit_ridge(Xb, y, lam=1e-6)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            Xb_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
            pred = Xb_test @ w
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
    
    return predict
# EVOLVE-BLOCK-END
