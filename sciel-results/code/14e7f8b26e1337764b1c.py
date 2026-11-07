# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.optimize import minimize
from numpy.polynomial import chebyshev as C

_TOTAL_DEADLINE = time.time() + 325.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _fit_chebyshev(X, y, max_degree=40, ridge_values=None):
    """
    Fit using Chebyshev polynomial basis with L2 regularization (ridge).
    This is numerically stable for 1D interpolation of noiseless data.
    """
    try:
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-12:
            return None, np.inf
        
        # Map to [-1, 1]
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        
        if ridge_values is None:
            ridge_values = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
        
        best_predict = None
        best_mse = np.inf
        
        for ridge in ridge_values:
            if _remaining() < 2.0:
                break
            
            # Build Chebyshev basis up to max_degree
            # C.chebvander(x, n) gives Chebyshev polynomials T_0(x) ... T_n(x)
            F = C.chebvander(x_norm, max_degree)
            
            # Add ridge to diagonal
            A = F.T @ F + ridge * np.eye(max_degree + 1)
            b = F.T @ y
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pred = F @ w
            if not np.all(np.isfinite(pred)):
                continue
            
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w.copy()
                best_ridge = ridge
        
        if best_predict is None and best_mse < np.inf:
            # Create prediction function
            def predict(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                xp_norm = 2.0 * (xp - x_min) / (x_max - x_min) - 1.0
                Fp = C.chebvander(xp_norm, max_degree)
                preds = Fp @ best_w
                return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
            return _safe_predict(predict), best_mse
        
        return None, np.inf
    except Exception:
        return None, np.inf


def _fit_fourier_cheb(X, y, max_harmonics=30):
    """
    Hybrid: fit both Fourier and Chebyshev, return the better one.
    """
    # Chebyshev
    ch_pred, ch_mse = _fit_chebyshev(X, y, max_degree=30)
    
    # Fourier
    x = X[:, 0]
    best_fourier_predict = None
    best_fourier_mse = np.inf
    
    for freq_mult in [0.5, 1.0, 2.0, np.pi, 2*np.pi, 0.25, 4.0, 0.1, 0.75, 1.5, 3.0, 5.0]:
        if _remaining() < 2.0:
            break
        fundamental = freq_mult
        cols = [np.ones(X.shape[0])]
        for k in range(1, max_harmonics + 1):
            cols.append(np.sin(k * fundamental * x))
            cols.append(np.cos(k * fundamental * x))
        F = np.column_stack(cols)
        if not np.all(np.isfinite(F)):
            continue
        
        # Ridge solve
        ridge = 1e-8
        A = F.T @ F + ridge * np.eye(F.shape[1])
        b = F.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            continue
        mse = np.mean((pred - y) ** 2)
        if mse < best_fourier_mse:
            best_fourier_mse = mse
            fund_cached = fundamental
            max_harm_cached = max_harmonics
            w_cached = w
            best_fourier_predict = _safe_predict(lambda Xp, f=fund_cached, m=max_harm_cached, w=w_cached: 
                np.column_stack([np.ones(Xp.shape[0])] + [np.sin(k*f*Xp[:,0]) for k in range(1, m+1)] + [np.cos(k*f*Xp[:,0]) for k in range(1, m+1)]) @ w)
    
    # Return the better of the two
    if ch_pred is not None and ch_mse < np.inf:
        if best_fourier_predict is None or ch_mse <= best_fourier_mse:
            return ch_pred, ch_mse
        else:
            return best_fourier_predict, best_fourier_mse
    elif best_fourier_predict is not None:
        return best_fourier_predict, best_fourier_mse
    else:
        return None, np.inf


def _fit_smooth_spline(X, y):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = len(xs)
        if n < 10:
            return None, np.inf
        
        best_predict = None
        best_rmse = np.inf
        
        for s in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
            if _remaining() < 2.0:
                break
            try:
                spl = UnivariateSpline(xs, ys, s=s)
                pred = spl(xs)
                rmse = np.sqrt(np.mean((pred - ys) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = _safe_predict(lambda Xp, spl=spl: np.nan_to_num(spl(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12))
            except Exception:
                continue
        
        return best_predict, best_rmse
    except Exception:
        return None, np.inf


def _fit_gaussian_process(X, y):
    """
    Gaussian Process with RBF kernel and MLE hyperparameter optimization.
    """
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mu) / y_std
        
        def kernel(X1, X2, length_scale, variance):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            return variance * np.exp(-0.5 * sq_dist)
        
        def neg_log_likelihood(params):
            length_scale = np.exp(params[0])
            variance = np.exp(params[1])
            noise = np.exp(params[2])
            
            K = kernel(Xn, Xn, length_scale, variance) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                mse_term = y_norm @ alpha
                logdet = 2 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * (n * np.log(2 * np.pi) + logdet + mse_term)
                return nll
            except Exception:
                return 1e10
        
        best_params = None
        best_nll = np.inf
        inits = [
            np.array([0.0, 0.0, -5.0]),
            np.array([-0.5, 0.5, -5.0]),
            np.array([0.5, 0.5, -5.0]),
            np.array([-1.0, 1.0, -6.0]),
            np.array([0.0, 1.0, -4.0]),
            np.array([1.0, 0.0, -4.0]),
        ]
        
        for init in inits:
            if _remaining() < 3.0:
                break
            result = minimize(neg_log_likelihood, init, method='Nelder-Mead', options={'maxiter': 100})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        
        if best_params is None:
            return None, np.inf
        
        length_scale = np.exp(best_params[0])
        variance = np.exp(best_params[1])
        noise = np.exp(best_params[2])
        
        K = kernel(Xn, Xn, length_scale, variance) + noise * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, length_scale, variance)
            preds = (Ks @ alpha).flatten() * y_std + y_mu
            return np.nan_to_num(preds, nan=y_mu, posinf=1e12, neginf=-1e12)
        
        pred_train = predict(X)
        rmse = np.sqrt(np.mean((pred_train - y) ** 2))
        if not np.isfinite(rmse):
            return None, np.inf
        return _safe_predict(predict), rmse
    except Exception:
        return None, np.inf


def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-6):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        w = np.linalg.solve(K, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            Ks = np.exp(-0.5 * sq_dist)
            preds = Ks @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return _safe_predict(predict)
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        n, d = X_train.shape

    # Hold-out validation split for model selection
    np.random.seed(12345)
    indices = np.random.permutation(n)
    val_frac = 0.2
    n_val = max(20, int(n * val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_fit, y_fit = X_train[train_idx], y_train[train_idx]
    
    best_predict = None
    best_val_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_fit)
    const_pred_val = np.full(len(y_val), const)
    const_val_rmse = np.sqrt(np.mean((const_pred_val - y_val) ** 2))
    best_val_rmse = const_val_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # 1. Try Chebyshev spectral collocation (primary structural change)
    if _remaining() > 10.0 and d == 1:
        for max_deg in [10, 15, 20, 25, 30, 35, 40]:
            if _remaining() < 5.0:
                break
            ch_pred, ch_mse = _fit_chebyshev(X_fit, y_fit, max_degree=max_deg)
            if ch_pred is not None and ch_mse < np.inf:
                val_pred = ch_pred(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_predict = ch_pred

    # 2. Try Fourier-Chebyshev hybrid
    if _remaining() > 10.0 and d == 1:
        fh_pred, fh_mse = _fit_fourier_cheb(X_fit, y_fit, max_harmonics=25)
        if fh_pred is not None and fh_mse < np.inf:
            val_pred = fh_pred(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = fh_pred

    # 3. Try smooth splines
    if _remaining() > 10.0 and d == 1:
        ss_pred, ss_mse = _fit_smooth_spline(X_fit, y_fit)
        if ss_pred is not None and ss_mse < np.inf:
            val_pred = ss_pred(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = ss_pred

    # 4. Try Gaussian Process
    if _remaining() > 20.0:
        gp_pred, gp_mse = _fit_gaussian_process(X_fit, y_fit)
        if gp_pred is not None and gp_mse < np.inf:
            val_pred = gp_pred(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = gp_pred

    # 5. Try Kernel Ridge Regression
    if _remaining() > 15.0:
        for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if _remaining() < 3.0:
                break
            kr_pred = _fit_kernel_ridge(X_fit, y_fit, length_scale=ls)
            if kr_pred is not None:
                val_pred = kr_pred(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_predict = kr_pred

    # 6. Refit best model family on ALL data if time permits
    if _remaining() > 20.0:
        candidates = []
        
        # Chebyshev on full data
        if d == 1:
            for max_deg in [20, 25, 30, 35, 40]:
                if _remaining() < 5.0:
                    break
                ch_full, _ = _fit_chebyshev(X_train, y_train, max_degree=max_deg)
                if ch_full is not None:
                    candidates.append(ch_full)
        
        # Fourier-Cheb on full data
        if d == 1 and _remaining() > 10.0:
            fh_full, _ = _fit_fourier_cheb(X_train, y_train, max_harmonics=25)
            if fh_full is not None:
                candidates.append(fh_full)
        
        # Smooth spline on full data
        if d == 1 and _remaining() > 10.0:
            ss_full, _ = _fit_smooth_spline(X_train, y_train)
            if ss_full is not None:
                candidates.append(ss_full)
        
        # GP on full data
        if _remaining() > 15.0:
            gp_full, _ = _fit_gaussian_process(X_train, y_train)
            if gp_full is not None:
                candidates.append(gp_full)
        
        # Kernel ridge on full data
        if _remaining() > 10.0:
            for ls in [0.3, 0.5, 1.0, 2.0]:
                if _remaining() < 3.0:
                    break
                kr_full = _fit_kernel_ridge(X_train, y_train, length_scale=ls)
                if kr_full is not None:
                    candidates.append(kr_full)
        
        # Evaluate candidates on validation set
        for cand in candidates:
            val_pred = cand(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_predict = cand

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
