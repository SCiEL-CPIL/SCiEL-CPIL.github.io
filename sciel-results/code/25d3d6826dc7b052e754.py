"""Symbolic regression via multi-strategy: interpolation, polynomial, GP, and feature-based regression."""
import time
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d, CubicSpline
from scipy.optimize import minimize_scalar

def _nrmse(y, yhat):
    """Compute normalized MSE."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-12:
        return mse
    return mse / vy

def _fit_polynomial(x, y, max_deg=12):
    """Fit polynomial and return best degree based on train score."""
    best_deg = 0
    best_score = np.inf
    best_coeff = None
    
    for deg in range(1, max_deg + 1):
        try:
            coeff = np.polyfit(x, y, deg)
            yhat = np.polyval(coeff, x)
            if not np.all(np.isfinite(yhat)):
                continue
            score = _nrmse(y, yhat)
            if score < best_score:
                best_score = score
                best_deg = deg
                best_coeff = coeff
        except Exception:
            continue
    
    return best_coeff, best_deg, best_score

def _fit_gp(x_train, y_train, x_test, length_scales=None, sigma=1.0):
    """Gaussian Process regression with RBF kernel."""
    if length_scales is None:
        length_scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    n = len(x_train)
    best_pred = None
    best_score = np.inf
    
    for ls in length_scales:
        try:
            K = np.exp(-0.5 * ((x_train[:, None] - x_train[None, :]) ** 2) / (ls ** 2)) * sigma ** 2
            K += 1e-8 * np.eye(n)
            
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            K_star = np.exp(-0.5 * ((x_test[:, None] - x_train[None, :]) ** 2) / (ls ** 2)) * sigma ** 2
            y_pred = K_star @ alpha
            
            if not np.all(np.isfinite(y_pred)):
                continue
            
            # Score on training data as proxy
            K_train_test = np.exp(-0.5 * ((x_train[:, None] - x_train[None, :]) ** 2) / (ls ** 2)) * sigma ** 2
            yhat_train = K_train_test @ alpha
            score = _nrmse(y_train, yhat_train)
            
            if score < best_score:
                best_score = score
                best_pred = y_pred
        except Exception:
            continue
    
    return best_pred, best_score

def _build_rich_features(X, max_poly=10):
    """Build a comprehensive feature set for 1D input."""
    n = X.shape[0]
    x = X[:, 0].astype(np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    
    # Polynomials
    x_pow = x.copy()
    for p in range(1, max_poly + 1):
        cols.append(x_pow)
        x_pow = x_pow * x
    
    # Absolute value based
    ax = np.abs(x)
    cols.append(ax)
    cols.append(np.sqrt(ax))
    cols.append(ax ** 0.25)
    cols.append(ax ** 0.75)
    cols.append(ax ** 1.25)
    cols.append(ax ** 1.5)
    cols.append(ax ** 2.0)
    
    # Signed powers
    for p in [0.5, 0.75, 1.25, 1.5, 2.5, 3.0]:
        cols.append(np.sign(x) * (ax ** p))
    
    # Reciprocals (guarded)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(np.abs(x) > 1e-8, 1.0 / x, 0.0)
        inv2 = np.where(np.abs(x) > 1e-8, 1.0 / (x * x), 0.0)
        inv_sqrt = np.where(ax > 1e-8, 1.0 / np.sqrt(ax), 0.0)
    cols.append(inv)
    cols.append(inv2)
    cols.append(inv_sqrt)
    
    # Trigonometric
    for freq in [1.0, 2.0, 0.5]:
        cols.append(np.sin(freq * x))
        cols.append(np.cos(freq * x))
        cols.append(np.sin(freq * x * x))
        cols.append(np.cos(freq * x * x))
    
    # Hyperbolic
    cols.append(np.tanh(x))
    cols.append(np.tanh(2 * x))
    cols.append(np.tanh(0.5 * x))
    
    # Exponential (clipped)
    xc = np.clip(x, -30, 30)
    cols.append(np.exp(xc))
    cols.append(np.exp(-xc))
    cols.append(np.exp(-x * x))
    cols.append(np.exp(-0.5 * x * x))
    cols.append(np.exp(-0.25 * x * x))
    
    # Logarithms (guarded)
    cols.append(np.where(x > 1e-9, np.log(x), 0.0))
    cols.append(np.log1p(ax))
    cols.append(np.where(x > 1e-9, np.log(x) * x, 0.0))
    
    # Mixed terms
    cols.append(x * np.sin(x))
    cols.append(x * np.cos(x))
    cols.append(x * np.tanh(x))
    cols.append(x * np.exp(-x * x))
    cols.append(ax * np.sin(x))
    cols.append(ax * np.cos(x))
    
    return np.column_stack(cols)

def _ridge_cv(Phi, y, lambdas, n_folds=5, seed=42):
    """Cross-validated ridge regression."""
    n = Phi.shape[0]
    n_feat = Phi.shape[1]
    
    if n_folds > n:
        n_folds = max(2, n // 10)
    
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    fold_size = n // n_folds
    if fold_size < 1:
        fold_size = 1
    
    best_lambda = lambdas[0]
    best_cv_score = np.inf
    best_w = None
    
    for lam in lambdas:
        cv_scores = []
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            if len(train_idx) < 2 or len(val_idx) < 1:
                continue
            
            Phi_tr = Phi[train_idx]
            y_tr = y[train_idx]
            Phi_val = Phi[val_idx]
            y_val = y[val_idx]
            
            try:
                if lam == 0.0:
                    w, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
                else:
                    A = Phi_tr.T @ Phi_tr + lam * np.eye(n_feat)
                    b = Phi_tr.T @ y_tr
                    L = np.linalg.cholesky(A)
                    w = np.linalg.solve(L.T, np.linalg.solve(L, b))
                
                yhat_val = Phi_val @ w
                if not np.all(np.isfinite(yhat_val)):
                    cv_scores.append(np.inf)
                    continue
                cv_scores.append(_nrmse(y_val, yhat_val))
            except Exception:
                cv_scores.append(np.inf)
        
        if cv_scores:
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_lambda = lam
        
        # Fit final model with best lambda so far
        if best_lambda == lam:
            try:
                if lam == 0.0:
                    best_w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                else:
                    A = Phi.T @ Phi + lam * np.eye(n_feat)
                    b = Phi.T @ y
                    L = np.linalg.cholesky(A)
                    best_w = np.linalg.solve(L.T, np.linalg.solve(L, b))
            except Exception:
                pass
    
    return best_w, best_lambda, best_cv_score

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 290.0
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    
    if d != 1:
        # Fallback for multi-dimensional (shouldn't happen per task)
        x = X_train.flatten()
        y = y_train
        coeff, deg, score = _fit_polynomial(x, y, max_deg=6)
        if coeff is not None:
            def predict_multi(X):
                xf = X.flatten()
                return np.polyval(coeff, xf)
            return predict_multi
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    x_train = X_train[:, 0].astype(np.float64)
    y_train_1d = y_train.astype(np.float64)
    
    # Sort by x
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train_1d[sort_idx]
    
    # Check for duplicates
    unique_x, inverse_idx = np.unique(x_sorted, return_inverse=True)
    has_duplicates = len(unique_x) < len(x_sorted)
    
    if has_duplicates:
        y_unique = np.array([y_sorted[inverse_idx == i].mean() for i in range(len(unique_x))])
        x_fit = unique_x
        y_fit = y_unique
    else:
        x_fit = x_sorted
        y_fit = y_sorted
    
    best_predict = None
    best_score = np.inf
    best_method = None
    
    # Strategy 1: Cubic spline interpolation (s=0 for noiseless)
    try:
        if len(x_fit) >= 4:
            spline = CubicSpline(x_fit, y_fit, bc_type='natural')
            yhat_train = spline(x_train)
            if np.all(np.isfinite(yhat_train)):
                train_score = _nrmse(y_train, yhat_train)
                if train_score < 1e-6:
                    best_score = train_score
                    best_method = 'cubic_spline'
                    def spline_predict(X):
                        x_test = X[:, 0]
                        yhat = spline(x_test)
                        return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                    best_predict = spline_predict
    except Exception:
        pass
    
    # Strategy 2: UnivariateSpline with s=0
    if best_score > 1e-5:
        try:
            if len(x_fit) >= 4:
                spline = UnivariateSpline(x_fit, y_fit, s=0, k=3)
                yhat_train = spline(x_train)
                if np.all(np.isfinite(yhat_train)):
                    train_score = _nrmse(y_train, yhat_train)
                    if train_score < best_score:
                        best_score = train_score
                        best_method = 'univariate_spline'
                        def spline2_predict(X):
                            x_test = X[:, 0]
                            yhat = spline(x_test)
                            return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                        best_predict = spline2_predict
        except Exception:
            pass
    
    # Strategy 3: Polynomial fit (try multiple degrees)
    if best_score > 1e-4:
        try:
            coeff, deg, poly_score = _fit_polynomial(x_fit, y_fit, max_deg=15)
            if coeff is not None and poly_score < best_score:
                best_score = poly_score
                best_method = f'poly_deg{deg}'
                coeff_final = coeff
                def poly_predict(X):
                    x_test = X[:, 0]
                    return np.polyval(coeff_final, x_test).astype(np.float64)
                best_predict = poly_predict
        except Exception:
            pass
    
    # Strategy 4: Linear interpolation (robust, exact on training points)
    if best_score > 1e-3:
        try:
            interp_func = interp1d(x_fit, y_fit, kind='linear', fill_value='extrapolate')
            yhat_train = interp_func(x_train)
            if np.all(np.isfinite(yhat_train)):
                train_score = _nrmse(y_train, yhat_train)
                if train_score < best_score:
                    best_score = train_score
                    best_method = 'linear_interp'
                    def linear_predict(X):
                        x_test = X[:, 0]
                        yhat = interp_func(x_test)
                        return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                    best_predict = linear_predict
        except Exception:
            pass
    
    # Strategy 5: Gaussian Process with multiple length scales
    if best_score > 1e-3 and time.time() < DEADLINE - 60:
        try:
            # Use a subset for GP to keep it fast
            n_gp = min(200, len(x_fit))
            if n_gp >= 10:
                # Use all points if not too many
                x_gp = x_fit
                y_gp = y_fit
                
                # Test GP on training data
                yhat_gp_train, gp_score = _fit_gp(x_gp, y_gp, x_gp, 
                                                   length_scales=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
                if yhat_gp_train is not None and gp_score < best_score:
                    # Precompute GP parameters
                    ls_list = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
                    best_ls = None
                    best_alpha = None
                    best_K_inv = None
                    best_gp_score = np.inf
                    
                    for ls in ls_list:
                        try:
                            K = np.exp(-0.5 * ((x_gp[:, None] - x_gp[None, :]) ** 2) / (ls ** 2))
                            K += 1e-8 * np.eye(len(x_gp))
                            L = np.linalg.cholesky(K)
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_gp))
                            yhat = (K @ alpha)
                            score = _nrmse(y_gp, yhat)
                            if score < best_gp_score:
                                best_gp_score = score
                                best_ls = ls
                                best_alpha = alpha
                        except Exception:
                            continue
                    
                    if best_ls is not None:
                        best_score = best_gp_score
                        best_method = f'gp_ls{best_ls}'
                        ls_final = best_ls
                        alpha_final = best_alpha
                        x_gp_final = x_gp
                        def gp_predict(X):
                            x_test = X[:, 0]
                            K_star = np.exp(-0.5 * ((x_test[:, None] - x_gp_final[None, :]) ** 2) / (ls_final ** 2))
                            yhat = K_star @ alpha_final
                            return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                        best_predict = gp_predict
        except Exception:
            pass
    
    # Strategy 6: Rich feature basis + CV Ridge
    if best_score > 1e-2 and time.time() < DEADLINE - 30:
        try:
            Phi = _build_rich_features(X_train, max_poly=10)
            
            # Clean features
            finite_mask = np.all(np.isfinite(Phi), axis=0)
            Phi_clean = Phi[:, finite_mask]
            col_var = np.var(Phi_clean, axis=0)
            keep_mask = col_var > 1e-10
            Phi_clean = Phi_clean[:, keep_mask]
            
            if Phi_clean.shape[1] > 0 and Phi_clean.shape[1] < n:
                lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
                w_final, lam_final, cv_score = _ridge_cv(Phi_clean, y_train, lambdas, n_folds=5)
                
                if w_final is not None:
                    yhat_train = Phi_clean @ w_final
                    if np.all(np.isfinite(yhat_train)):
                        train_score = _nrmse(y_train, yhat_train)
                        if train_score < best_score:
                            best_score = train_score
                            best_method = 'ridge_rich'
                            w_final_copy = w_final.copy()
                            keep_mask_copy = keep_mask.copy()
                            finite_mask_copy = finite_mask.copy()
                            def ridge_predict(X):
                                P = _build_rich_features(X, max_poly=10)
                                P = P[:, finite_mask_copy]
                                P = P[:, keep_mask_copy]
                                yhat = P @ w_final_copy
                                return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                            best_predict = ridge_predict
        except Exception:
            pass
    
    # Strategy 7: Try specific functional forms
    if best_score > 1e-2 and time.time() < DEADLINE - 20:
        # Try: a + b*x + c*sin(x) + d*cos(x) + e*exp(-x^2)
        try:
            x = x_fit
            y = y_fit
            A = np.column_stack([
                np.ones_like(x),
                x,
                np.sin(x),
                np.cos(x),
                np.exp(-x * x)
            ])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            yhat = A @ w
            if np.all(np.isfinite(yhat)):
                score = _nrmse(y, yhat)
                if score < best_score:
                    best_score = score
                    best_method = 'parametric_sin_cos_exp'
                    w_final = w
                    def param_predict(X):
                        xt = X[:, 0]
                        At = np.column_stack([
                            np.ones_like(xt),
                            xt,
                            np.sin(xt),
                            np.cos(xt),
                            np.exp(-xt * xt)
                        ])
                        return (At @ w_final).astype(np.float64)
                    best_predict = param_predict
        except Exception:
            pass
    
    # Final fallback
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    return best_predict
