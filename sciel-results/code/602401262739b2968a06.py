import numpy as np
import time
import warnings
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar, minimize

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

def _loocv_ridge_fast(K, y, alpha):
    """LOOCV for ridge regression in RKHS using the matrix inverse trick.
    K is n x n kernel matrix, y is n-vector, alpha is regularization.
    Returns LOOCV MSE."""
    n = len(y)
    K_reg = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(K_reg, y, rcond=None)
    # LOOCV: prediction for i is K[i,:] @ w - K[i,i] * w[i] / (1 + alpha * diag_inv)
    # Actually, for ridge: yhat_i_loo = y_i - K_ii / (1 + alpha * [K^{-1}]_ii) * residual_i
    # More precisely: yhat_loo = K @ w - diag(K) * w / (1 + alpha * diag((K+alpha*I)^{-1}))
    # Using: (K+alpha*I)^{-1} = (1/alpha)(I - K(K+alpha*I)^{-1})
    # Let's use the standard formula: yhat_loo_i = y_i - e_i^T (K+alpha*I)^{-1} K y / (1 - alpha * [(K+alpha*I)^{-1}]_ii)
    # Simpler: compute H = K @ (K + alpha*I)^{-1}, then yhat_loo = y - diag(H) / (1 - alpha*diag((K+alpha*I)^{-1})) * (y - H@y)
    # Actually the cleanest: yhat_loo = K @ w - K_ii * w_i / (1 + alpha * [inv(K+alpha*I)]_ii)
    # Let's just do it directly with the hat matrix approach
    try:
        K_inv = np.linalg.inv(K_reg)
    except np.linalg.LinAlgError:
        return np.inf
    # Hat matrix H = K @ (K + alpha*I)^{-1}
    H = K @ K_inv
    # LOOCV predictions: yhat_loo_i = (H @ y)_i - H_ii * y_i / (1 - H_ii)
    Hy = H @ y
    Hdiag = np.diag(H)
    denom = 1.0 - Hdiag
    # Avoid division by zero
    mask = np.abs(denom) > 1e-12
    yhat_loo = np.copy(Hy)
    yhat_loo[mask] = (Hy[mask] - Hdiag[mask] * y[mask]) / denom[mask]
    yhat_loo[~mask] = Hy[~mask]
    mse = np.mean((yhat_loo - y)**2)
    return mse

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
    """Cross-validate RBF kernel + ridge using LOOCV for better selection."""
    n = len(y)
    
    best_mse = np.inf
    best_ls = length_scales[0]
    best_alpha = alphas[0]
    
    for ls in length_scales:
        if _time_left() < 5.0:
            break
        K = _rbf_kernel(x, x, ls)
        for alpha in alphas:
            if _time_left() < 5.0:
                break
            try:
                loocv_mse = _loocv_ridge_fast(K, y, alpha)
            except Exception:
                loocv_mse = np.inf
            
            if loocv_mse < best_mse:
                best_mse = loocv_mse
                best_ls = ls
                best_alpha = alpha
    
    return best_ls, best_alpha, best_mse

def _build_chebyshev_basis(x, x_min, x_max, n_terms):
    """Build Chebyshev polynomial basis of degree 0 to n_terms-1."""
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    T = np.zeros((len(x), n_terms))
    if n_terms > 0:
        T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = x_norm
    for k in range(2, n_terms):
        T[:, k] = 2 * x_norm * T[:, k-1] - T[:, k-2]
    return T

def _build_rich_basis(x, x_min, x_max, n_poly, n_fourier, n_exp):
    """Build a rich basis combining polynomials, Fourier, and exponential terms."""
    n = len(x)
    total = n_poly + 2 * n_fourier + n_exp
    B = np.zeros((n, total))
    
    # Polynomial terms (Chebyshev)
    if n_poly > 0:
        B[:, :n_poly] = _build_chebyshev_basis(x, x_min, x_max, n_poly)
    
    # Fourier terms
    L = x_max - x_min
    if L <= 0:
        L = 1.0
    col = n_poly
    for k in range(1, n_fourier + 1):
        arg = k * np.pi * (x - x_min) / L
        if col < total:
            B[:, col] = np.cos(arg)
            col += 1
        if col < total:
            B[:, col] = np.sin(arg)
            col += 1
    
    # Exponential terms
    col = n_poly + 2 * n_fourier
    for k in range(n_exp):
        if col < total:
            rate = (k + 1) * 0.5 / L
            B[:, col] = np.exp(rate * (x - x_min))
            col += 1
    
    return B

def _fit_rich_basis_ridge(x, y, x_test, n_poly, n_fourier, n_exp, lam, x_min, x_max):
    """Fit rich basis + ridge."""
    B = _build_rich_basis(x, x_min, x_max, n_poly, n_fourier, n_exp)
    n_terms = B.shape[1]
    A = B.T @ B + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, B.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, B.T @ y, rcond=None)
    B_test = _build_rich_basis(x_test, x_min, x_max, n_poly, n_fourier, n_exp)
    return B_test @ w

def _fit_rich_basis_cv(x, y, configs, lambdas, n_folds=5):
    """Cross-validate rich basis + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_config = configs[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    
    for config in configs:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_rich_basis_ridge(x[train_idx], y[train_idx], x[val_idx], 
                                                     config[0], config[1], config[2], lam, x_min, x_max)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_config = config
                best_lambda = lam
    
    return best_config, best_lambda, best_mse

def _fit_chebyshev_ridge(x, y, x_test, n_terms, lam, x_min, x_max):
    """Fit Chebyshev polynomial + ridge."""
    V = _build_chebyshev_basis(x, x_min, x_max, n_terms)
    A = V.T @ V + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, V.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
    V_test = _build_chebyshev_basis(x_test, x_min, x_max, n_terms)
    return V_test @ w

def _fit_chebyshev_ridge_cv(x, y, n_terms_options, lambdas, n_folds=5):
    """Cross-validate Chebyshev basis + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_n = n_terms_options[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    
    for n_terms in n_terms_options:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_chebyshev_ridge(x[train_idx], y[train_idx], x[val_idx], n_terms, lam, x_min, x_max)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_n = n_terms
                best_lambda = lam
    
    return best_n, best_lambda, best_mse

def _build_fourier_basis(x, x_min, x_max, n_terms):
    """Build Fourier basis: 1, cos(k*pi*x/L), sin(k*pi*x/L) for k=1.."""
    L = x_max - x_min
    if L <= 0:
        L = 1.0
    n = len(x)
    # n_terms total columns: 1 + 2*floor((n_terms-1)/2)
    F = np.zeros((n, n_terms))
    F[:, 0] = 1.0
    col = 1
    k = 1
    while col < n_terms:
        arg = k * np.pi * (x - x_min) / L
        F[:, col] = np.cos(arg)
        col += 1
        if col < n_terms:
            F[:, col] = np.sin(arg)
            col += 1
        k += 1
    return F

def _fit_fourier_ridge(x, y, x_test, n_terms, lam, x_min, x_max):
    """Fit Fourier basis + ridge."""
    F = _build_fourier_basis(x, x_min, x_max, n_terms)
    A = F.T @ F + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, F.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, F.T @ y, rcond=None)
    F_test = _build_fourier_basis(x_test, x_min, x_max, n_terms)
    return F_test @ w

def _fit_fourier_ridge_cv(x, y, n_terms_options, lambdas, n_folds=5):
    """Cross-validate Fourier basis + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_n = n_terms_options[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    
    for n_terms in n_terms_options:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_fourier_ridge(x[train_idx], y[train_idx], x[val_idx], n_terms, lam, x_min, x_max)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_n = n_terms
                best_lambda = lam
    
    return best_n, best_lambda, best_mse

def _spline_fit(x, y, x_test, s):
    """Fit a smoothing spline."""
    try:
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        
        if s <= 0:
            s = 1e-8 * max(np.var(y), 1e-10)
        
        spl = UnivariateSpline(x_sorted, y_sorted, k=3, s=s)
        return spl(x_test)
    except Exception:
        return np.full(len(x_test), np.mean(y))

def _spline_cv(x, y, s_options, n_folds=5):
    """Cross-validate smoothing spline."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_s = s_options[0]
    
    for s in s_options:
        if _time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            try:
                pred_val = _spline_fit(x[train_idx], y[train_idx], x[val_idx], s)
                mse = np.mean((pred_val - y[val_idx])**2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_s = s
    
    return best_s, best_mse

def _fit_cubic_spline(x, y, x_test):
    """Fit an interpolating cubic spline (no smoothing)."""
    try:
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        spl = CubicSpline(x_sorted, y_sorted)
        return spl(x_test)
    except Exception:
        return np.full(len(x_test), np.mean(y))

def _fit_loess(x_train, y_train, x_test, frac, degree=1):
    """LOESS prediction."""
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    
    n = len(x_train)
    m = max(2, int(frac * n))
    preds = np.empty(len(x_test))
    
    for i, xv in enumerate(x_test):
        if _time_left() < 0.5:
            preds[i:] = np.mean(y_train)
            break
        
        dists = np.abs(x_train - xv)
        if m >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(dists, m - 1)[:m]
        
        max_dist = dists[idx].max()
        if max_dist < 1e-15:
            max_dist = 1e-10
        w = (1 - (dists[idx] / max_dist)**3)**3
        
        # Weighted polynomial fit
        try:
            if degree == 0:
                preds[i] = np.sum(w * y_train[idx]) / np.sum(w)
            else:
                Xw = x_train[idx] - xv
                W = np.diag(w)
                A = np.vstack([np.ones_like(Xw), Xw]).T
                WA = W @ A
                Wb = W @ y_train[idx]
                try:
                    coef = np.linalg.solve(WA.T @ A + 1e-12 * np.eye(2), WA.T @ y_train[idx])
                    preds[i] = coef[0]
                except np.linalg.LinAlgError:
                    preds[i] = np.sum(w * y_train[idx]) / np.sum(w)
        except Exception:
            preds[i] = np.mean(y_train)
    
    return preds

def _loess_cv(x, y, frac_options, n_folds=5):
    """Cross-validate LOESS."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_frac = frac_options[0]
    
    for frac in frac_options:
        if _time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            try:
                pred_val = _fit_loess(x[train_idx], y[train_idx], x[val_idx], frac)
                mse = np.mean((pred_val - y[val_idx])**2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_frac = frac
    
    return best_frac, best_mse

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
    
    x_min, x_max = np.min(x), np.max(x)
    
    results = []
    
    # Method 1: RBF kernel + ridge (LOOCV)
    if _time_left() > 30.0:
        ls_options = np.logspace(-2, 1.5, 10)
        alpha_options = np.logspace(-5, 1, 8)
        best_ls, best_alpha, kernel_mse = _fit_rbf_kernel_ridge_cv(x, y, ls_options, alpha_options)
        results.append(('kernel', kernel_mse, best_ls, best_alpha))
    
    # Method 2: Chebyshev basis + ridge
    if _time_left() > 30.0:
        n_terms_options = [5, 8, 12, 16, 20, 25, 30]
        lam_options = np.logspace(-4, 2, 10)
        best_n_cheb, best_lam_cheb, cheb_mse = _fit_chebyshev_ridge_cv(x, y, n_terms_options, lam_options)
        results.append(('chebyshev', cheb_mse, best_n_cheb, best_lam_cheb))
    
    # Method 3: Fourier basis + ridge
    if _time_left() > 30.0:
        n_terms_fourier = [5, 8, 12, 16, 20, 25]
        best_n_four, best_lam_four, four_mse = _fit_fourier_ridge_cv(x, y, n_terms_fourier, lam_options)
        results.append(('fourier', four_mse, best_n_four, best_lam_four))
    
    # Method 4: Rich basis (poly + fourier + exp) + ridge
    if _time_left() > 40.0:
        rich_configs = [
            (10, 3, 3),
            (12, 5, 3),
            (15, 5, 5),
            (10, 5, 5),
            (8, 3, 5),
            (12, 3, 5),
        ]
        rich_lams = np.logspace(-4, 2, 8)
        best_rich_config, best_rich_lam, rich_mse = _fit_rich_basis_cv(x, y, rich_configs, rich_lams)
        results.append(('rich', rich_mse, best_rich_config, best_rich_lam))
    
    # Method 5: Smoothing spline
    if _time_left() > 20.0:
        s_options = np.logspace(-12, -2, 20)
        best_s, spline_mse = _spline_cv(x, y, s_options)
        results.append(('spline', spline_mse, best_s, None))
    
    # Method 6: LOESS
    if _time_left() > 15.0:
        frac_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        best_frac, loess_mse = _loess_cv(x, y, frac_options)
        results.append(('loess', loess_mse, best_frac, None))
    
    # Method 7: Interpolating cubic spline
    if _time_left() > 10.0:
        try:
            np.random.seed(42)
            indices = np.arange(n)
            np.random.shuffle(indices)
            folds = np.array_split(indices, 5)
            fold_mses = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                pred_val = _fit_cubic_spline(x[train_idx], y[train_idx], x[val_idx])
                mse = np.mean((pred_val - y[val_idx])**2)
                fold_mses.append(mse)
            spline_interp_mse = np.mean(fold_mses)
            results.append(('spline_interp', spline_interp_mse, None, None))
        except Exception:
            pass
    
    # If no methods succeeded, fall back to mean
    if not results:
        mean_y = np.mean(y)
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.full(len(X_arr), mean_y)
        return predict
    
    # Pick the best method
    results.sort(key=lambda m: m[1])
    best_method = results[0][0]
    
    # Build the final model
    if best_method == 'kernel':
        _, _, best_ls, best_alpha = results[0]
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
    
    elif best_method == 'chebyshev':
        _, _, best_n_cheb, best_lam_cheb = results[0]
        V = _build_chebyshev_basis(x, x_min, x_max, best_n_cheb)
        A = V.T @ V + best_lam_cheb * np.eye(best_n_cheb)
        try:
            w = np.linalg.solve(A, V.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
        
        w_store = w.copy()
        xmin_store = x_min
        xmax_store = x_max
        n_store = best_n_cheb
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            V_test = _build_chebyshev_basis(xv, xmin_store, xmax_store, n_store)
            return V_test @ w_store
    
    elif best_method == 'fourier':
        _, _, best_n_four, best_lam_four = results[0]
        F = _build_fourier_basis(x, x_min, x_max, best_n_four)
        A = F.T @ F + best_lam_four * np.eye(best_n_four)
        try:
            w = np.linalg.solve(A, F.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, F.T @ y, rcond=None)
        
        w_store = w.copy()
        xmin_store = x_min
        xmax_store = x_max
        n_store = best_n_four
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            F_test = _build_fourier_basis(xv, xmin_store, xmax_store, n_store)
            return F_test @ w_store
    
    elif best_method == 'rich':
        _, _, best_config, best_lam = results[0]
        n_poly, n_fourier, n_exp = best_config
        B = _build_rich_basis(x, x_min, x_max, n_poly, n_fourier, n_exp)
        n_terms = B.shape[1]
        A = B.T @ B + best_lam * np.eye(n_terms)
        try:
            w = np.linalg.solve(A, B.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, B.T @ y, rcond=None)
        
        w_store = w.copy()
        xmin_store = x_min
        xmax_store = x_max
        config_store = best_config
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            B_test = _build_rich_basis(xv, xmin_store, xmax_store, config_store[0], config_store[1], config_store[2])
            return B_test @ w_store
    
    elif best_method == 'spline':
        _, _, best_s, _ = results[0]
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        try:
            spl = UnivariateSpline(x_sorted, y_sorted, k=3, s=best_s)
            
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = X_arr[:, 0]
                return spl(xv)
        except Exception:
            mean_y = np.mean(y)
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return np.full(len(X_arr), mean_y)
    
    elif best_method == 'loess':
        _, _, best_frac, _ = results[0]
        x_store = x.copy()
        y_store = y.copy()
        frac_store = best_frac
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            return _fit_loess(x_store, y_store, xv, frac_store)
    
    else:  # spline_interp
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        try:
            spl = CubicSpline(x_sorted, y_sorted)
            
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = X_arr[:, 0]
                return spl(xv)
        except Exception:
            mean_y = np.mean(y)
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return np.full(len(X_arr), mean_y)
    
    return predict
