"""Symbolic regression using kernel ridge regression with CV selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix."""
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    d2 = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-gamma * d2)


def _poly_kernel(X1, X2, degree, coef0=1.0):
    """Compute polynomial kernel matrix."""
    return (coef0 + X1 @ X2.T) ** degree


def _sigmoid_kernel(X1, X2, gamma, coef0=0.0):
    """Compute sigmoid kernel matrix."""
    return np.tanh(gamma * X1 @ X2.T + coef0)


def _fourier_features(x, n_freqs=20, scale=1.0):
    """Random Fourier features for RBF approximation."""
    n = len(x)
    # Use deterministic frequencies based on input range
    x_min, x_max = np.min(x), np.max(x)
    span = max(x_max - x_min, 1e-6)
    # Frequencies from 0 to ~1/span for good coverage
    freqs = np.linspace(0.1, 5.0 / span, n_freqs)
    cols = [np.ones(n)]
    for f in freqs:
        cols.append(np.sin(f * x))
        cols.append(np.cos(f * x))
    return np.column_stack(cols)


def _cv_score(X, y, kernel_fn, n_folds=5):
    """Compute mean CV MSE using kernel ridge regression."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    fold_size = n // n_folds
    mse_sum = 0.0
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        K_tr = kernel_fn(X_tr, X_tr)
        K_val = kernel_fn(X_val, X_tr)
        
        # Ridge: (K + alpha*I)^{-1} y
        # We need to pick alpha; use a fixed reasonable value for CV
        alpha = 1e-3
        K_reg = K_tr + alpha * np.eye(len(X_tr))
        try:
            w = np.linalg.solve(K_reg, y_tr)
            pred = K_val @ w
            mse = np.mean((pred - y_val) ** 2)
            if not np.isfinite(mse):
                return np.inf
            mse_sum += mse
        except np.linalg.LinAlgError:
            return np.inf
    
    return mse_sum / n_folds


def _fit_kernel_ridge(X, y, kernel_fn, alpha=1e-3):
    """Fit kernel ridge regression, return predict function."""
    n = len(y)
    K = kernel_fn(X, X)
    K_reg = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        K_new = kernel_fn(X_new, X)
        return K_new @ w
    
    return predict


def _fit_linear_ridge(X, y, alpha=1e-3):
    """Fit linear ridge regression."""
    n, d = X.shape
    A = np.column_stack([np.ones(n), X])
    AtA = A.T @ A + alpha * np.eye(A.shape[1])
    Aty = A.T @ y
    try:
        w = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        A_new = np.column_stack([np.ones(len(X_new)), X_new])
        return A_new @ w
    
    return predict


def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    # Normalize y for better conditioning
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    # For 1D input, use kernel methods with CV
    if d == 1:
        x = X_train[:, 0]
        x_min, x_max = np.min(x), np.max(x)
        span = max(x_max - x_min, 1e-6)
        
        # Candidate kernels with different hyperparameters
        candidates = []
        
        # RBF kernels with various bandwidths
        for gamma in [0.1 / span, 0.5 / span, 1.0 / span, 2.0 / span, 5.0 / span, 10.0 / span]:
            candidates.append(('rbf', gamma))
        
        # Polynomial kernels
        for degree in [2, 3, 4, 5, 6, 8]:
            candidates.append(('poly', degree))
        
        # Sigmoid kernels
        for gamma in [0.5 / span, 1.0 / span, 2.0 / span]:
            candidates.append(('sigmoid', gamma))
        
        # Fourier features (linear model on Fourier basis)
        candidates.append(('fourier', 20))
        candidates.append(('fourier', 40))
        
        # Linear
        candidates.append(('linear', 0))
        
        best_predict = None
        best_cv_score = np.inf
        
        for name, param in candidates:
            if _time_left() < 10:
                break
            
            try:
                if name == 'rbf':
                    gamma = param
                    kf = lambda X1, X2, g=gamma: _rbf_kernel(X1, X2, g)
                    cv_score = _cv_score(X_train, y_norm, kf)
                elif name == 'poly':
                    degree = param
                    kf = lambda X1, X2, d=degree: _poly_kernel(X1, X2, d)
                    cv_score = _cv_score(X_train, y_norm, kf)
                elif name == 'sigmoid':
                    gamma = param
                    kf = lambda X1, X2, g=gamma: _sigmoid_kernel(X1, X2, g)
                    cv_score = _cv_score(X_train, y_norm, kf)
                elif name == 'fourier':
                    n_freqs = param
                    # For Fourier, we use linear model on Fourier features
                    # CV: fit linear ridge on Fourier features
                    def fourier_cv(X, y):
                        n = len(y)
                        indices = np.arange(n)
                        np.random.seed(42)
                        indices = np.random.permutation(indices)
                        fold_size = n // 5
                        mse_sum = 0.0
                        for fold in range(5):
                            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                            tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                            X_tr = X[tr_idx]
                            y_tr = y[tr_idx]
                            X_val = X[val_idx]
                            y_val = y[val_idx]
                            F_tr = _fourier_features(X_tr, n_freqs=n_freqs)
                            F_val = _fourier_features(X_val, n_freqs=n_freqs)
                            alpha = 1e-3
                            AtA = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
                            Aty = F_tr.T @ y_tr
                            try:
                                w = np.linalg.solve(AtA, Aty)
                                pred = F_val @ w
                                mse = np.mean((pred - y_val) ** 2)
                                if not np.isfinite(mse):
                                    return np.inf
                                mse_sum += mse
                            except np.linalg.LinAlgError:
                                return np.inf
                        return mse_sum / 5
                    cv_score = fourier_cv(X_train, y_norm)
                elif name == 'linear':
                    def linear_cv(X, y):
                        n = len(y)
                        indices = np.arange(n)
                        np.random.seed(42)
                        indices = np.random.permutation(indices)
                        fold_size = n // 5
                        mse_sum = 0.0
                        for fold in range(5):
                            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                            tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                            X_tr = X[tr_idx]
                            y_tr = y[tr_idx]
                            X_val = X[val_idx]
                            y_val = y[val_idx]
                            A_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
                            A_val = np.column_stack([np.ones(len(X_val)), X_val])
                            alpha = 1e-3
                            AtA = A_tr.T @ A_tr + alpha * np.eye(A_tr.shape[1])
                            Aty = A_tr.T @ y_tr
                            try:
                                w = np.linalg.solve(AtA, Aty)
                                pred = A_val @ w
                                mse = np.mean((pred - y_val) ** 2)
                                if not np.isfinite(mse):
                                    return np.inf
                                mse_sum += mse
                            except np.linalg.LinAlgError:
                                return np.inf
                        return mse_sum / 5
                    cv_score = linear_cv(X_train, y_norm)
                else:
                    continue
                
                if cv_score < best_cv_score and np.isfinite(cv_score):
                    best_cv_score = cv_score
                    
                    # Fit on all data with best hyperparameter
                    if name == 'rbf':
                        kf = lambda X1, X2, g=param: _rbf_kernel(X1, X2, g)
                        best_predict = _fit_kernel_ridge(X_train, y_norm, kf, alpha=1e-3)
                    elif name == 'poly':
                        kf = lambda X1, X2, d=param: _poly_kernel(X1, X2, d)
                        best_predict = _fit_kernel_ridge(X_train, y_norm, kf, alpha=1e-3)
                    elif name == 'sigmoid':
                        kf = lambda X1, X2, g=param: _sigmoid_kernel(X1, X2, g)
                        best_predict = _fit_kernel_ridge(X_train, y_norm, kf, alpha=1e-3)
                    elif name == 'fourier':
                        F = _fourier_features(x, n_freqs=param)
                        alpha = 1e-3
                        AtA = F.T @ F + alpha * np.eye(F.shape[1])
                        Aty = F.T @ y_norm
                        try:
                            w = np.linalg.solve(AtA, Aty)
                        except np.linalg.LinAlgError:
                            w = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
                        def predict(X_new, F_w=w, nf=param):
                            X_new = np.asarray(X_new, dtype=np.float64)
                            x_new = X_new[:, 0] if X_new.ndim > 1 else X_new
                            F_new = _fourier_features(x_new, n_freqs=nf)
                            return F_new @ F_w
                        best_predict = predict
                    elif name == 'linear':
                        best_predict = _fit_linear_ridge(X_train, y_norm, alpha=1e-3)
                        
            except Exception:
                continue
        
        # If no kernel worked, fallback to simple polynomial
        if best_predict is None:
            A = np.column_stack([np.ones(n), x, x**2, x**3, x**4])
            w, *_ = np.linalg.lstsq(A, y_norm, rcond=None)
            def predict(X_new):
                X_new = np.asarray(X_new, dtype=np.float64)
                x_new = X_new[:, 0] if X_new.ndim > 1 else X_new
                A_new = np.column_stack([np.ones(len(x_new)), x_new, x_new**2, x_new**3, x_new**4])
                return A_new @ w
            best_predict = predict
        
        # Wrap to denormalize
        def predict_final(X):
            pred_norm = best_predict(X)
            return pred_norm * y_std + y_mean
        
        return predict_final
    
    else:
        # Multi-dimensional: use kernel ridge with RBF
        x_min = np.min(X_train, axis=0)
        x_max = np.max(X_train, axis=0)
        span = np.max(x_max - x_min)
        if span < 1e-6:
            span = 1.0
        
        candidates = []
        for gamma in [0.1 / span, 0.5 / span, 1.0 / span, 2.0 / span, 5.0 / span]:
            candidates.append(('rbf', gamma))
        for degree in [2, 3, 4]:
            candidates.append(('poly', degree))
        candidates.append(('linear', 0))
        
        best_predict = None
        best_cv_score = np.inf
        
        for name, param in candidates:
            if _time_left() < 10:
                break
            
            try:
                if name == 'rbf':
                    gamma = param
                    kf = lambda X1, X2, g=gamma: _rbf_kernel(X1, X2, g)
                    cv_score = _cv_score(X_train, y_norm, kf)
                elif name == 'poly':
                    degree = param
                    kf = lambda X1, X2, d=degree: _poly_kernel(X1, X2, d)
                    cv_score = _cv_score(X_train, y_norm, kf)
                elif name == 'linear':
                    def linear_cv(X, y):
                        n = len(y)
                        indices = np.arange(n)
                        np.random.seed(42)
                        indices = np.random.permutation(indices)
                        fold_size = n // 5
                        mse_sum = 0.0
                        for fold in range(5):
                            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                            tr_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                            X_tr = X[tr_idx]
                            y_tr = y[tr_idx]
                            X_val = X[val_idx]
                            y_val = y[val_idx]
                            A_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
                            A_val = np.column_stack([np.ones(len(X_val)), X_val])
                            alpha = 1e-3
                            AtA = A_tr.T @ A_tr + alpha * np.eye(A_tr.shape[1])
                            Aty = A_tr.T @ y_tr
                            try:
                                w = np.linalg.solve(AtA, Aty)
                                pred = A_val @ w
                                mse = np.mean((pred - y_val) ** 2)
                                if not np.isfinite(mse):
                                    return np.inf
                                mse_sum += mse
                            except np.linalg.LinAlgError:
                                return np.inf
                        return mse_sum / 5
                    cv_score = linear_cv(X_train, y_norm)
                else:
                    continue
                
                if cv_score < best_cv_score and np.isfinite(cv_score):
                    best_cv_score = cv_score
                    if name == 'rbf':
                        kf = lambda X1, X2, g=param: _rbf_kernel(X1, X2, g)
                        best_predict = _fit_kernel_ridge(X_train, y_norm, kf, alpha=1e-3)
                    elif name == 'poly':
                        kf = lambda X1, X2, d=param: _poly_kernel(X1, X2, d)
                        best_predict = _fit_kernel_ridge(X_train, y_norm, kf, alpha=1e-3)
                    elif name == 'linear':
                        best_predict = _fit_linear_ridge(X_train, y_norm, alpha=1e-3)
                        
            except Exception:
                continue
        
        if best_predict is None:
            A = np.column_stack([np.ones(n), X_train])
            w, *_ = np.linalg.lstsq(A, y_norm, rcond=None)
            def predict(X_new):
                X_new = np.asarray(X_new, dtype=np.float64)
                A_new = np.column_stack([np.ones(len(X_new)), X_new])
                return A_new @ w
            best_predict = predict
        
        def predict_final(X):
            pred_norm = best_predict(X)
            return pred_norm * y_std + y_mean
        
        return predict_final
# EVOLVE-BLOCK-END
