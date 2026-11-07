"""Hybrid regression: LOESS with LOOCV + kernel ridge + basis portfolio."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _rbf_kernel(X1, X2, gamma):
    sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sqdist = np.maximum(sqdist, 0)
    return np.exp(-gamma * sqdist)

def _loess_predict_at(x_new, x_train, y_train, frac, degree=2):
    """LOESS prediction at a single point x_new."""
    n = len(x_train)
    k = max(3, int(frac * n))
    
    # Find k nearest neighbors
    dists = np.abs(x_new - x_train)
    if k >= n:
        idx = np.arange(n)
        weights = np.ones(n)
    else:
        idx = np.argsort(dists)[:k]
        max_dist = dists[idx[-1]] + 1e-12
        weights = np.clip(1.0 - (dists[idx] / max_dist)**3, 0, 1)
        weights = weights ** 2  # Tricube-like
    
    # Local weighted polynomial regression
    x_loc = x_train[idx]
    y_loc = y_train[idx]
    
    # Build local design matrix
    cols = [np.ones(len(x_loc))]
    for d in range(1, degree + 1):
        cols.append((x_loc - x_new) ** d)
    X_loc = np.column_stack(cols)
    
    # Weighted least squares
    W = np.diag(weights)
    try:
        XtWX = X_loc.T @ W @ X_loc
        XtWy = X_loc.T @ W @ y_loc
        # Add small ridge for stability
        XtWX += 1e-8 * np.eye(degree + 1)
        w = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    
    # Predict at x_new (offset is 0)
    pred = w[0]
    return pred

def _loess_predict_vectorized(x_new_arr, x_train, y_train, frac, degree=2):
    """LOESS prediction for multiple points. Uses vectorization where possible."""
    x_new_arr = np.asarray(x_new_arr).flatten()
    n_new = len(x_new_arr)
    n = len(x_train)
    k = max(3, int(frac * n))
    
    preds = np.zeros(n_new)
    
    for i, xi in enumerate(x_new_arr):
        dists = np.abs(xi - x_train)
        if k >= n:
            idx = np.arange(n)
            weights = np.ones(n)
        else:
            idx = np.argsort(dists)[:k]
            max_dist = dists[idx[-1]] + 1e-12
            weights = np.clip(1.0 - (dists[idx] / max_dist)**3, 0, 1)
            weights = weights ** 2
        
        x_loc = x_train[idx]
        y_loc = y_train[idx]
        
        cols = [np.ones(len(x_loc))]
        for d in range(1, degree + 1):
            cols.append((x_loc - xi) ** d)
        X_loc = np.column_stack(cols)
        
        W = np.diag(weights)
        try:
            XtWX = X_loc.T @ W @ X_loc
            XtWy = X_loc.T @ W @ y_loc
            XtWX += 1e-8 * np.eye(degree + 1)
            w = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        
        preds[i] = w[0]
    
    return preds

def _loocv_loess(x_train, y_train, fracs, degree=2):
    """Leave-one-out cross-validation for LOESS to select best frac."""
    n = len(x_train)
    best_frac = fracs[0]
    best_cv_mse = float('inf')
    
    for frac in fracs:
        cv_preds = np.zeros(n)
        for i in range(n):
            if _time_left() < 10.0:
                return best_frac, best_cv_mse
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_lo = x_train[mask]
            y_lo = y_train[mask]
            cv_preds[i] = _loess_predict_at(x_train[i], x_lo, y_lo, frac, degree)
        
        cv_mse = np.mean((cv_preds - y_train)**2)
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_frac = frac
    
    return best_frac, best_cv_mse

def _fit_loess(X_train, y_train):
    """Fit LOESS with LOOCV-selected bandwidth. Returns (train_mse, predict_func)."""
    x = X_train[:, 0]
    y = y_train
    
    # Candidate bandwidths (fraction of data)
    fracs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_frac, best_cv_mse = _loocv_loess(x, y, fracs, degree=2)
    
    # Final prediction function
    def predict(X):
        x_new = X[:, 0]
        return _loess_predict_vectorized(x_new, x, y, best_frac, degree=2)
    
    # Compute training MSE for comparison
    train_pred = _loess_predict_vectorized(x, x, y, best_frac, degree=2)
    train_mse = np.mean((train_pred - y)**2)
    
    return train_mse, predict

def _fit_kernel_ridge(X, y, kernel_type, params):
    """Fit kernel ridge regression. Returns (predict_func, train_mse)."""
    n = X.shape[0]
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    Xn = (X - X_mean) / X_std
    y_mean = y.mean()
    yn = y - y_mean

    if kernel_type == 'rbf':
        gamma = params.get('gamma', 1.0)
        K = _rbf_kernel(Xn, Xn, gamma)
    elif kernel_type == 'linear':
        K = Xn @ Xn.T
    elif kernel_type == 'poly':
        degree = params.get('degree', 3)
        K = (Xn @ Xn.T + 1) ** degree
    else:
        return None, np.inf

    lam = params.get('lam', 1e-3)
    K_reg = K + lam * np.eye(n)

    try:
        alpha = np.linalg.solve(K_reg, yn)
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.lstsq(K_reg, yn, rcond=None)[0]
        except:
            return None, np.inf

    train_pred = K @ alpha + y_mean
    train_mse = np.mean((train_pred - y) ** 2)

    def predict(X_test):
        Xn_test = (X_test - X_mean) / X_std
        if kernel_type == 'rbf':
            K_test = _rbf_kernel(Xn_test, Xn, gamma)
        elif kernel_type == 'linear':
            K_test = Xn_test @ Xn.T
        elif kernel_type == 'poly':
            degree = params.get('degree', 3)
            K_test = (Xn_test @ Xn.T + 1) ** degree
        else:
            return np.zeros(len(X_test))
        return K_test @ alpha + y_mean

    return predict, train_mse

def _fit_basis(X, y, basis_type, params):
    """Fit basis function regression. Returns (predict_func, train_mse)."""
    n, d = X.shape
    x = X[:, 0]

    if basis_type == 'poly':
        deg = params.get('deg', 5)
        cols = [np.ones(n)]
        for k in range(1, deg + 1):
            cols.append(x ** k)
        feat = np.column_stack(cols)
    elif basis_type == 'trig':
        max_k = params.get('max_k', 6)
        cols = [np.ones(n)]
        for k in range(1, max_k + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        feat = np.column_stack(cols)
    elif basis_type == 'exp_log':
        x_clip = np.clip(x, -10, 10)
        cols = [np.ones(n), np.exp(x_clip), np.exp(-x_clip)]
        safe_x = np.where(x > 0, x, 1.0)
        cols.append(np.log(safe_x))
        cols.append(np.log1p(np.abs(x)))
        feat = np.column_stack(cols)
    elif basis_type == 'rational':
        safe_x = np.where(np.abs(x) > 1e-6, x, 1e-6)
        feat = np.column_stack([np.ones(n), x, 1.0 / safe_x, x ** 2, x ** 3])
    elif basis_type == 'tanh':
        feat = np.column_stack([np.ones(n), np.tanh(x), np.tanh(2 * x), np.tanh(0.5 * x)])
    elif basis_type == 'hinge':
        thresholds = params.get('thresholds', [-2, -1, 0, 1, 2])
        cols = [np.ones(n)]
        for t in thresholds:
            cols.append(np.maximum(0, x - t))
            cols.append(np.maximum(0, t - x))
        feat = np.column_stack(cols)
    else:
        return None, np.inf

    lam = params.get('lam', 1e-4)
    XtX = feat.T @ feat
    Xty = feat.T @ y
    A = XtX + lam * np.eye(feat.shape[1])
    try:
        w = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Xty, rcond=None)[0]

    train_pred = feat @ w
    train_mse = np.mean((train_pred - y) ** 2)

    def predict(X_test):
        n_test = X_test.shape[0]
        x_test = X_test[:, 0]
        if basis_type == 'poly':
            cols = [np.ones(n_test)]
            for k in range(1, deg + 1):
                cols.append(x_test ** k)
            feat_test = np.column_stack(cols)
        elif basis_type == 'trig':
            cols = [np.ones(n_test)]
            for k in range(1, max_k + 1):
                cols.append(np.sin(k * x_test))
                cols.append(np.cos(k * x_test))
            feat_test = np.column_stack(cols)
        elif basis_type == 'exp_log':
            x_clip = np.clip(x_test, -10, 10)
            cols = [np.ones(n_test), np.exp(x_clip), np.exp(-x_clip)]
            safe_x = np.where(x_test > 0, x_test, 1.0)
            cols.append(np.log(safe_x))
            cols.append(np.log1p(np.abs(x_test)))
            feat_test = np.column_stack(cols)
        elif basis_type == 'rational':
            safe_x = np.where(np.abs(x_test) > 1e-6, x_test, 1e-6)
            feat_test = np.column_stack([np.ones(n_test), x_test, 1.0 / safe_x, x_test ** 2, x_test ** 3])
        elif basis_type == 'tanh':
            feat_test = np.column_stack([np.ones(n_test), np.tanh(x_test), np.tanh(2 * x_test), np.tanh(0.5 * x_test)])
        elif basis_type == 'hinge':
            cols = [np.ones(n_test)]
            for t in thresholds:
                cols.append(np.maximum(0, x_test - t))
                cols.append(np.maximum(0, t - x_test))
            feat_test = np.column_stack(cols)
        else:
            return np.zeros(n_test)
        return feat_test @ w

    return predict, train_mse

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    d = X_train.shape[1]
    if d != 1:
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

    candidates = []

    # LOESS with LOOCV - primary structural change
    candidates.append(('loess', None))

    # Kernel ridge regression candidates
    for gamma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
            candidates.append(('kernel', 'rbf', {'gamma': gamma, 'lam': lam}))

    for degree in [2, 3, 4, 5]:
        for lam in [1e-4, 1e-3, 1e-2]:
            candidates.append(('kernel', 'poly', {'degree': degree, 'lam': lam}))

    for lam in [1e-4, 1e-3, 1e-2]:
        candidates.append(('kernel', 'linear', {'lam': lam}))

    # Basis function candidates
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        for lam in [0.0, 1e-6, 1e-4, 1e-2]:
            candidates.append(('basis', 'poly', {'deg': deg, 'lam': lam}))

    for max_k in [3, 4, 5, 6, 8]:
        for lam in [0.0, 1e-6, 1e-4]:
            candidates.append(('basis', 'trig', {'max_k': max_k, 'lam': lam}))

    for lam in [0.0, 1e-4, 1e-2]:
        candidates.append(('basis', 'exp_log', {'lam': lam}))
        candidates.append(('basis', 'rational', {'lam': lam}))
        candidates.append(('basis', 'tanh', {'lam': lam}))
        candidates.append(('basis', 'hinge', {'thresholds': [-2, -1, 0, 1, 2], 'lam': lam}))

    best_mse = float('inf')
    best_predict = None

    for model_type, func_type, params in candidates:
        if _time_left() < 5.0:
            break

        try:
            if model_type == 'loess':
                train_mse, predict_func = _fit_loess(X_train, y_train)
            elif model_type == 'kernel':
                predict_func, train_mse = _fit_kernel_ridge(X_train, y_train, func_type, params)
            else:
                predict_func, train_mse = _fit_basis(X_train, y_train, func_type, params)

            if predict_func is not None and np.isfinite(train_mse) and train_mse < best_mse:
                best_mse = train_mse
                best_predict = predict_func
        except Exception:
            continue

    if best_predict is None:
        Xf = np.column_stack([np.ones(len(X_train)), X_train[:, 0], X_train[:, 0]**2])
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        best_predict = lambda X, w=w: np.column_stack([np.ones(len(X)), X[:, 0], X[:, 0]**2]) @ w

    return best_predict
# EVOLVE-BLOCK-END
