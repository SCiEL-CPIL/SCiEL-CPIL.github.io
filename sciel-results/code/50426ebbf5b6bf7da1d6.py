import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 30.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _loess_predict(x_train, y_train, x_test, frac, degree=1):
    """LOESS (locally weighted scatterplot smoothing) prediction."""
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    
    n = len(x_train)
    m = max(1, int(frac * n))
    preds = np.empty(len(x_test))
    
    for i, xv in enumerate(x_test):
        if _time_left() < 0.5:
            preds[i:] = np.mean(y_train)
            break
        
        # Find m nearest neighbors
        dists = np.abs(x_train - xv)
        if m >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(dists, m)[:m]
        
        # Tricube weights
        max_dist = dists[idx].max()
        if max_dist < 1e-12:
            w = np.ones(len(idx))
        else:
            u = dists[idx] / max_dist
            w = (1 - u**3)**3
            w[w < 1e-12] = 0
        
        # Weighted polynomial fit
        Xw = np.column_stack([np.ones(len(idx)), x_train[idx]])
        if degree >= 2:
            Xw = np.column_stack([Xw, (x_train[idx] - xv)**2])
        
        W = np.diag(w)
        try:
            beta = np.linalg.solve(Xw.T @ W @ Xw, Xw.T @ W @ y_train[idx])
            xv_row = np.array([1.0, xv])
            if degree >= 2:
                xv_row = np.append(xv_row, (xv - xv)**2)
            preds[i] = xv_row @ beta
        except np.linalg.LinAlgError:
            preds[i] = np.average(y_train[idx], weights=w) if w.sum() > 0 else np.mean(y_train)
    
    return preds

def _loocv_loess(x, y, fracs, degree=1):
    """Leave-one-out cross-validation for LOESS."""
    n = len(y)
    best_frac = fracs[0]
    best_mse = np.inf
    
    for frac in fracs:
        if _time_left() < 2.0:
            break
        
        # LOOCV: predict each point using all others
        m = max(1, int(frac * n))
        loocv_errors = np.zeros(n)
        
        for i in range(n):
            if _time_left() < 0.1:
                break
            # Leave out point i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_tr = x[mask]
            y_tr = y[mask]
            
            # Predict point i
            dists = np.abs(x_tr - x[i])
            if m >= len(x_tr):
                idx = np.arange(len(x_tr))
            else:
                idx = np.argpartition(dists, m)[:m]
            
            max_dist = dists[idx].max()
            if max_dist < 1e-12:
                w = np.ones(len(idx))
            else:
                u = dists[idx] / max_dist
                w = (1 - u**3)**3
                w[w < 1e-12] = 0
            
            Xw = np.column_stack([np.ones(len(idx)), x_tr[idx]])
            if degree >= 2:
                Xw = np.column_stack([Xw, (x_tr[idx] - x[i])**2])
            
            W = np.diag(w)
            try:
                beta = np.linalg.solve(Xw.T @ W @ Xw, Xw.T @ W @ y_tr[idx])
                pred = np.array([1.0, x[i]]) @ beta if degree < 2 else \
                       np.array([1.0, x[i], 0.0]) @ beta
                loocv_errors[i] = (pred - y[i])**2
            except np.linalg.LinAlgError:
                loocv_errors[i] = (np.average(y_tr[idx], weights=w) - y[i])**2 if w.sum() > 0 else (np.mean(y_tr) - y[i])**2
        
        mse = np.mean(loocv_errors[:n])
        if mse < best_mse:
            best_mse = mse
            best_frac = frac
    
    return best_frac, best_mse

def _rbf_kernel(X1, X2, length_scale):
    X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=np.float64).reshape(-1, 1)
    d2 = np.sum((X1 - X2.T)**2, axis=1)
    return np.exp(-0.5 * d2 / (length_scale**2))

def _fit_ridge_kernel(x_train, y_train, x_test, length_scale, alpha):
    """Fit ridge regression in RKHS induced by RBF kernel."""
    K = _rbf_kernel(x_train, x_train, length_scale)
    n = len(x_train)
    K_reg = K + alpha * np.eye(n)
    K_test = _rbf_kernel(x_test, x_train, length_scale)
    
    try:
        w = np.linalg.solve(K_reg, y_train)
        return K_test @ w
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(K_reg, y_train, rcond=None)
        return K_test @ w

def _fit_rbf_kernel_ridge_cv(x, y, length_scales, alphas, n_folds=5):
    """Cross-validate RBF kernel + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_ls = length_scales[0]
    best_alpha = alphas[0]
    
    for ls in length_scales:
        if _time_left() < 5.0:
            break
        for alpha in alphas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_ridge_kernel(x[train_idx], y[train_idx], x[val_idx], ls, alpha)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_ls = ls
                best_alpha = alpha
    
    return best_ls, best_alpha, best_mse

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(y)
    x = X[:, 0].copy()
    
    # Method 1: LOESS with LOOCV
    frac_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    best_frac, loess_mse = _loocv_loess(x, y, frac_options, degree=1)
    
    # Method 2: RBF kernel + ridge
    ls_options = np.logspace(-2, 1, 8)
    alpha_options = np.logspace(-4, 1, 8)
    best_ls, best_alpha, kernel_mse = _fit_rbf_kernel_ridge_cv(x, y, ls_options, alpha_options)
    
    # Pick the best method
    if loess_mse <= kernel_mse:
        method = 'loess'
    else:
        method = 'kernel'
    
    # Build the final model
    if method == 'loess':
        x_store = x.copy()
        y_store = y.copy()
        frac_store = best_frac
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            return _loess_predict(x_store, y_store, xv, frac_store, degree=1)
    
    else:
        K = _rbf_kernel(x, x, best_ls)
        K_reg = K + best_alpha * np.eye(n)
        try:
            w = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(K_reg, y, rcond=None)
        
        x_store = x.copy()
        w_store = w.copy()
        ls_store = best_ls
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            K_test = _rbf_kernel(xv, x_store, ls_store)
            return K_test @ w_store
    
    return predict
