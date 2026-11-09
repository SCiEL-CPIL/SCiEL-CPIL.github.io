"""Combined symbolic regression: Interpolation-first with GP/Thin-plate splines + CV Ridge fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import norm

def _nrmse(y, yhat):
    """Compute normalized MSE."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-12:
        return mse
    return mse / vy

def _build_features(X, max_p=8):
    """Build a rich feature matrix from input X (n, d)."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n, dtype=np.float64)]
    
    for j in range(d):
        x = X[:, j].astype(np.float64)
        ax = np.abs(x)
        
        # Polynomials x^1 .. x^max_p
        powv = x.copy()
        for p in range(1, max_p + 1):
            cols.append(powv)
            if p < max_p:
                powv = powv * x
        
        # Abs and sqrt of abs
        cols.append(ax)
        sqrt_ax = np.sqrt(ax)
        cols.append(sqrt_ax)
        cols.append(sqrt_ax * x)
        cols.append(ax ** 0.25)
        
        # Reciprocal-ish (guarded)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(np.abs(x) > 1e-6, 1.0 / x, 0.0)
        cols.append(inv)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv2 = np.where(np.abs(x) > 1e-6, 1.0 / (x * x), 0.0)
        cols.append(inv2)
        
        # Trigonometric
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.sin(x * x))
        cols.append(np.cos(x * x))
        
        # Exponential (clipped)
        xc = np.clip(x, -50, 50)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        cols.append(np.exp(-x * x))
        
        # Logarithms (guarded)
        cols.append(np.where(x > 1e-9, np.log(x), 0.0))
        cols.append(np.log1p(np.abs(x)))
        
        # Power laws
        cols.append(x * np.sqrt(ax))
        cols.append(np.sign(x) * (ax ** 0.75))
        cols.append(np.sign(x) * (ax ** 1.25))
        cols.append(np.sign(x) * (ax ** 0.5))
        cols.append(np.sign(x) * (ax ** 1.5))
    
    # Pairwise interactions
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)

def _clean_features(Phi):
    """Remove non-finite and low-variance columns."""
    finite_cols = np.all(np.isfinite(Phi), axis=0)
    Phi_clean = Phi[:, finite_cols]
    if Phi_clean.shape[1] == 0:
        return Phi, np.zeros(Phi.shape[1], dtype=bool)
    col_var = np.var(Phi_clean, axis=0)
    keep = col_var > 1e-12
    return Phi_clean[:, keep], keep

def _ridge_solve(Phi, y, lam):
    """Solve ridge regression for given lambda."""
    nfeat = Phi.shape[1]
    if lam == 0.0:
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    else:
        A = Phi.T @ Phi + lam * np.eye(nfeat)
        b = Phi.T @ y
        try:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L.T, np.linalg.solve(L, b))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _fit_gp_interpolation(x_train, y_train, x_test, length_scale=1.0, sigma=1.0):
    """Simple Gaussian Process interpolation for 1D noiseless data."""
    # For noiseless data, use exact interpolation with a kernel
    # RBF kernel: k(x, x') = sigma^2 * exp(-||x-x'||^2 / (2*l^2))
    n = len(x_train)
    K = np.exp(-0.5 * ((x_train[:, None] - x_train[None, :]) ** 2) / (length_scale ** 2)) * sigma ** 2
    K += 1e-10 * np.eye(n)  # small jitter for numerical stability
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        K_star = np.exp(-0.5 * ((x_test[:, None] - x_train[None, :]) ** 2) / (length_scale ** 2)) * sigma ** 2
        y_pred = K_star @ alpha
        return y_pred
    except np.linalg.LinAlgError:
        return None

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 290.0
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    y_mean = float(np.mean(y_train))
    
    # Strategy: For 1D noiseless data, try interpolation methods first
    # These can achieve near-zero error if the function is smooth
    
    best_predict = None
    best_score = np.inf
    best_method = None
    
    if d == 1:
        x_train = X_train[:, 0]
        y_train_1d = y_train
        
        # Sort by x
        sort_idx = np.argsort(x_train)
        x_sorted = x_train[sort_idx]
        y_sorted = y_train_1d[sort_idx]
        
        # Check for duplicate x values
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        has_duplicates = len(unique_x) < len(x_sorted)
        
        # Method 1: Cubic spline interpolation (s=0 for noiseless)
        try:
            if not has_duplicates:
                spline = UnivariateSpline(x_sorted, y_sorted, s=0, k=3)
                
                # Validate on training data
                yhat_train = spline(x_train)
                train_score = _nrmse(y_train, yhat_train)
                
                if np.all(np.isfinite(yhat_train)) and train_score < 1e-6:
                    best_score = train_score
                    best_method = 'spline'
                    best_predict = lambda X: np.where(
                        np.isfinite(spline(X[:, 0])), 
                        spline(X[:, 0]), 
                        y_mean
                    ).astype(np.float64)
            else:
                # Handle duplicates by averaging y values
                unique_x, inverse = np.unique(x_sorted, return_inverse=True)
                y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
                spline = UnivariateSpline(unique_x, y_unique, s=0, k=3)
                
                yhat_train = spline(x_train)
                train_score = _nrmse(y_train, yhat_train)
                
                if np.all(np.isfinite(yhat_train)) and train_score < 1e-6:
                    best_score = train_score
                    best_method = 'spline_dup'
                    best_predict = lambda X: np.where(
                        np.isfinite(spline(X[:, 0])), 
                        spline(X[:, 0]), 
                        y_mean
                    ).astype(np.float64)
        except Exception:
            pass
        
        # Method 2: Linear interpolation (robust fallback)
        if best_score > 1e-4:
            try:
                if not has_duplicates:
                    interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                    yhat_train = interp_func(x_train)
                    train_score = _nrmse(y_train, yhat_train)
                    
                    if np.all(np.isfinite(yhat_train)) and train_score < best_score:
                        best_score = train_score
                        best_method = 'linear_interp'
                        best_predict = lambda X: np.where(
                            np.isfinite(interp_func(X[:, 0])), 
                            interp_func(X[:, 0]), 
                            y_mean
                        ).astype(np.float64)
                else:
                    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
                    y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
                    interp_func = interp1d(unique_x, y_unique, kind='linear', fill_value='extrapolate')
                    yhat_train = interp_func(x_train)
                    train_score = _nrmse(y_train, yhat_train)
                    
                    if np.all(np.isfinite(yhat_train)) and train_score < best_score:
                        best_score = train_score
                        best_method = 'linear_interp_dup'
                        best_predict = lambda X: np.where(
                            np.isfinite(interp_func(X[:, 0])), 
                            interp_func(X[:, 0]), 
                            y_mean
                        ).astype(np.float64)
            except Exception:
                pass
        
        # Method 3: GP interpolation with multiple length scales
        if best_score > 1e-4:
            for ls in [0.5, 1.0, 2.0, 5.0]:
                if time.time() > DEADLINE - 20:
                    break
                try:
                    # Test GP on training data first
                    yhat_gp = _fit_gp_interpolation(x_sorted, y_sorted, x_sorted, length_scale=ls, sigma=1.0)
                    if yhat_gp is not None and np.all(np.isfinite(yhat_gp)):
                        train_score = _nrmse(y_sorted, yhat_gp)
                        if train_score < best_score:
                            best_score = train_score
                            best_method = f'gp_ls{ls}'
                            # Precompute for prediction
                            x_tr = x_sorted
                            y_tr = y_sorted
                            ls_val = ls
                            K_train = np.exp(-0.5 * ((x_tr[:, None] - x_tr[None, :]) ** 2) / (ls_val ** 2))
                            K_train += 1e-10 * np.eye(len(x_tr))
                            L = np.linalg.cholesky(K_train)
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                            
                            def gp_predict(X, x_tr=x_tr, alpha=alpha, ls_val=ls_val):
                                x_test = X[:, 0]
                                K_star = np.exp(-0.5 * ((x_test[:, None] - x_tr[None, :]) ** 2) / (ls_val ** 2))
                                y_pred = K_star @ alpha
                                return np.where(np.isfinite(y_pred), y_pred, y_mean).astype(np.float64)
                            
                            best_predict = gp_predict
                except Exception:
                    continue
        
        # Method 4: Rich basis + CV Ridge (fallback)
        if best_score > 1e-3:
            try:
                Phi = _build_features(X_train, max_p=8)
                Phi_clean, keep_mask = _clean_features(Phi)
                
                if Phi_clean.shape[1] > 0:
                    best_w = None
                    best_cv_score = np.inf
                    best_lambda = None
                    
                    lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
                    
                    n_folds = 5
                    fold_size = n // n_folds
                    indices = np.arange(n)
                    np.random.seed(42)
                    np.random.shuffle(indices)
                    
                    for lam in lambdas:
                        if time.time() > DEADLINE - 10:
                            break
                        cv_scores = []
                        try:
                            for fold in range(n_folds):
                                val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
                                train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                                
                                Phi_tr = Phi_clean[train_idx]
                                y_tr = y_train[train_idx]
                                Phi_val = Phi_clean[val_idx]
                                y_val = y_train[val_idx]
                                
                                w = _ridge_solve(Phi_tr, y_tr, lam)
                                yhat_val = Phi_val @ w
                                if not np.all(np.isfinite(yhat_val)):
                                    cv_scores.append(np.inf)
                                    continue
                                cv_scores.append(_nrmse(y_val, yhat_val))
                            
                            mean_cv = np.mean(cv_scores)
                            if mean_cv < best_cv_score:
                                best_cv_score = mean_cv
                                best_lambda = lam
                        except Exception:
                            continue
                    
                    if best_lambda is not None:
                        best_w = _ridge_solve(Phi_clean, y_train, best_lambda)
                        best_w = np.where(np.isfinite(best_w), best_w, 0.0)
                        
                        yhat_train = Phi_clean @ best_w
                        yhat_train = np.where(np.isfinite(yhat_train), yhat_train, 0.0)
                        train_score = _nrmse(y_train, yhat_train)
                        
                        if train_score < best_score:
                            best_score = train_score
                            best_method = 'ridge'
                            w_final = best_w
                            mask_final = keep_mask
                            
                            def ridge_predict(X, w=w_final, mask=mask_final):
                                P = _build_features(X, max_p=8)
                                P_clean = P[:, mask]
                                yhat = P_clean @ w
                                return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                            
                            best_predict = ridge_predict
            except Exception:
                pass
    
    # Multi-dimensional fallback: CV Ridge
    if best_predict is None or d > 1:
        try:
            Phi = _build_features(X_train, max_p=8)
            Phi_clean, keep_mask = _clean_features(Phi)
            
            if Phi_clean.shape[1] > 0:
                best_w = None
                best_cv_score = np.inf
                best_lambda = None
                
                lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
                
                n_folds = 5
                fold_size = n // n_folds
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                
                for lam in lambdas:
                    if time.time() > DEADLINE - 10:
                        break
                    cv_scores = []
                    try:
                        for fold in range(n_folds):
                            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
                            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                            
                            Phi_tr = Phi_clean[train_idx]
                            y_tr = y_train[train_idx]
                            Phi_val = Phi_clean[val_idx]
                            y_val = y_train[val_idx]
                            
                            w = _ridge_solve(Phi_tr, y_tr, lam)
                            yhat_val = Phi_val @ w
                            if not np.all(np.isfinite(yhat_val)):
                                cv_scores.append(np.inf)
                                continue
                            cv_scores.append(_nrmse(y_val, yhat_val))
                        
                        mean_cv = np.mean(cv_scores)
                        if mean_cv < best_cv_score:
                            best_cv_score = mean_cv
                            best_lambda = lam
                    except Exception:
                        continue
                
                if best_lambda is not None:
                    best_w = _ridge_solve(Phi_clean, y_train, best_lambda)
                    best_w = np.where(np.isfinite(best_w), best_w, 0.0)
                    
                    yhat_train = Phi_clean @ best_w
                    yhat_train = np.where(np.isfinite(yhat_train), yhat_train, 0.0)
                    train_score = _nrmse(y_train, yhat_train)
                    
                    if best_predict is None or train_score < best_score:
                        w_final = best_w
                        mask_final = keep_mask
                        
                        def ridge_predict(X, w=w_final, mask=mask_final):
                            P = _build_features(X, max_p=8)
                            P_clean = P[:, mask]
                            yhat = P_clean @ w
                            return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                        
                        best_predict = ridge_predict
                        best_score = train_score
                        best_method = 'ridge_multi'
        except Exception:
            pass
    
    # Final fallback: mean
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    return best_predict

# EVOLVE-BLOCK-END
