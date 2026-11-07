"""
Symbolic regression over 4 hidden 1D equations.
Strategy:
  - Use a very rich finite basis (polys, trig, tanh, exp, log, rational, splines).
  - Normalize x for numerical stability.
  - Use a small, fast CV to pick regularization for ridge.
  - Also try a few direct least-squares models (polynomial degrees, RBF with a few gammas).
  - Pick the model with the best held-out MSE.
  - Fallback to a cubic polynomial if anything fails.
  - Keep an explicit deadline guard: return best-so-far before the limit.
"""
import numpy as np
import time
import warnings
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore")

_GLOBAL_START = None
TIME_LIMIT = 315.0  # 330s kill; leave 15s buffer

def _elapsed():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return time.time() - _GLOBAL_START

def _time_left():
    return TIME_LIMIT - _elapsed()

def _basis(x):
    """
    x: 1D array (normalized)
    Returns (N, P) basis matrix.
    """
    N = len(x)
    feats = [np.ones(N)]
    # Polynomials up to degree 10
    for k in range(1, 11):
        feats.append(x ** k)
    # Trig with several periods
    for T in [0.5, 1.0, 1.5, 2.0, np.pi, 2 * np.pi, 4 * np.pi]:
        feats.append(np.sin(x / T))
        feats.append(np.cos(x / T))
    # Tanh
    for a in [0.5, 1.0, 2.0]:
        feats.append(np.tanh(a * x))
    # Exponentials (bounded)
    for a in [-0.2, -0.5, -1.0, 0.2, 0.5]:
        feats.append(np.exp(a * x))
    # Gaussian bumps
    for mu in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for sigma in [0.5, 1.0, 2.0]:
            feats.append(np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)))
    # Log and rational
    feats.append(np.log1p(np.abs(x)))
    feats.append(np.sign(x) * np.log1p(np.abs(x)))
    feats.append(1.0 / (1.0 + x ** 2))
    feats.append(1.0 / (1.0 + 0.5 * x ** 2))
    feats.append(1.0 / (1.0 + x ** 2 + x))
    feats.append(np.sin(x) * np.cos(x))
    feats.append(np.sin(x) ** 2)
    feats.append(np.cos(x) ** 2)
    
    M = np.column_stack(feats)
    M = np.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    return M

def _ridge_fit(X, y, lam):
    N, P = X.shape
    A = X.T @ X + lam * np.eye(P)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _cv_ridge(X, y, lams, n_folds=4):
    N = len(y)
    rng = np.random.RandomState(0)
    idx = np.arange(N)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    best_lam = lams[0]
    best_mse = np.inf
    for lam in lams:
        mses = []
        for i in range(n_folds):
            val = folds[i]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            w = _ridge_fit(X[tr], y[tr], lam)
            pred = X[val] @ w
            mse = np.mean((pred - y[val]) ** 2)
            if not np.isfinite(mse):
                mse = np.inf
            mses.append(mse)
        m = np.mean(mses)
        if m < best_mse:
            best_mse = m
            best_lam = lam
    return best_lam, best_mse

def _cv_poly(x, y, degrees, n_folds=4):
    N = len(y)
    rng = np.random.RandomState(0)
    idx = np.arange(N)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    best_deg = degrees[0]
    best_mse = np.inf
    for deg in degrees:
        mses = []
        for i in range(n_folds):
            val = folds[i]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            Xtr = np.vander(x[tr], N=deg + 1, increasing=True)
            try:
                w = np.linalg.lstsq(Xtr, y[tr], rcond=None)[0]
                pred = np.vander(x[val], N=deg + 1, increasing=True) @ w
                mse = np.mean((pred - y[val]) ** 2)
                if not np.isfinite(mse):
                    mse = np.inf
                mses.append(mse)
            except:
                mses.append(np.inf)
        m = np.mean(mses)
        if m < best_mse:
            best_mse = m
            best_deg = deg
    return best_deg, best_mse

def _fit_and_predict(x, y, x_mean, x_std):
    """
    Try multiple model families, pick best by CV.
    Returns predict callable.
    """
    N = len(y)
    x_norm = (x - x_mean) / x_std
    
    candidates = []  # list of (predict_func, cv_score)
    
    # 1. Ridge on rich basis
    if _time_left() > 30:
        try:
            Xb = _basis(x_norm)
            lams = np.logspace(-3, 3, 30)
            best_lam, cv_mse = _cv_ridge(Xb, y, lams, n_folds=4)
            if np.isfinite(cv_mse):
                w = _ridge_fit(Xb, y, best_lam)
                if np.all(np.isfinite(w)):
                    def predict_ridge(X_test):
                        X_t = np.asarray(X_test, dtype=float)
                        x_t = X_t[:, 0]
                        x_tn = (x_t - x_mean) / x_std
                        Xb_t = _basis(x_tn)
                        return Xb_t @ w
                    candidates.append((predict_ridge, cv_mse))
        except:
            pass
    
    # 2. Polynomial via vander
    if _time_left() > 20:
        try:
            degrees = [2, 3, 4, 5, 6, 7, 8]
            best_deg, cv_mse = _cv_poly(x_norm, y, degrees, n_folds=4)
            if np.isfinite(cv_mse):
                Xv = np.vander(x_norm, N=best_deg + 1, increasing=True)
                w = np.linalg.lstsq(Xv, y, rcond=None)[0]
                if np.all(np.isfinite(w)):
                    def predict_poly(X_test):
                        X_t = np.asarray(X_test, dtype=float)
                        x_t = X_t[:, 0]
                        x_tn = (x_t - x_mean) / x_std
                        Xv_t = np.vander(x_tn, N=best_deg + 1, increasing=True)
                        return Xv_t @ w
                    candidates.append((predict_poly, cv_mse))
        except:
            pass
    
    # 3. Spline
    if _time_left() > 15:
        try:
            s_grid = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
            best_s = None
            best_mse = np.inf
            n_folds = 4
            rng = np.random.RandomState(0)
            idx = np.arange(N)
            rng.shuffle(idx)
            folds = np.array_split(idx, n_folds)
            for s in s_grid:
                mses = []
                for i in range(n_folds):
                    val = folds[i]
                    tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                    try:
                        sp = UnivariateSpline(x[tr], y[tr], s=s, k=4)
                        pred = sp(x[val])
                        mse = np.mean((pred - y[val]) ** 2)
                        if not np.isfinite(mse):
                            mse = np.inf
                        mses.append(mse)
                    except:
                        mses.append(np.inf)
                m = np.mean(mses)
                if m < best_mse:
                    best_mse = m
                    best_s = s
            if best_s is not None and np.isfinite(best_mse):
                sort_idx = np.argsort(x)
                sp = UnivariateSpline(x[sort_idx], y[sort_idx], s=best_s, k=4)
                def predict_spline(X_test):
                    X_t = np.asarray(X_test, dtype=float)
                    x_t = X_t[:, 0]
                    return sp(x_t)
                candidates.append((predict_spline, best_mse))
        except:
            pass
    
    # 4. RBF with a few gammas
    if _time_left() > 10:
        try:
            gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
            best_g = None
            best_mse = np.inf
            n_folds = 4
            rng = np.random.RandomState(0)
            idx = np.arange(N)
            rng.shuffle(idx)
            folds = np.array_split(idx, n_folds)
            for g in gammas:
                mses = []
                for i in range(n_folds):
                    val = folds[i]
                    tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                    try:
                        Xtr = x[tr][:, None]
                        Xval = x[val][:, None]
                        K = np.exp(-g * (Xtr - Xval) ** 2)
                        lam = 1e-3
                        w = np.linalg.solve(K + lam * np.eye(len(tr)), y[tr])
                        pred = K @ w
                        mse = np.mean((pred - y[val]) ** 2)
                        if not np.isfinite(mse):
                            mse = np.inf
                        mses.append(mse)
                    except:
                        mses.append(np.inf)
                m = np.mean(mses)
                if m < best_mse:
                    best_mse = m
                    best_g = g
            if best_g is not None and np.isfinite(best_mse):
                Xall = x[:, None]
                Kall = np.exp(-best_g * (Xall - Xall) ** 2)
                lam = 1e-3
                w = np.linalg.solve(Kall + lam * np.eye(N), y)
                Xtrain = x
                def predict_rbf(X_test):
                    X_t = np.asarray(X_test, dtype=float)
                    x_t = X_t[:, 0]
                    Kt = np.exp(-best_g * (x_t[:, None] - Xtrain[None, :]) ** 2)
                    return Kt @ w
                candidates.append((predict_rbf, best_mse))
        except:
            pass
    
    # 5. Fallback: cubic polynomial
    if len(candidates) == 0:
        Xv = np.vander(x_norm, N=4, increasing=True)
        w = np.linalg.lstsq(Xv, y, rcond=None)[0]
        def predict_fallback(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            x_tn = (x_t - x_mean) / x_std
            Xv_t = np.vander(x_tn, N=4, increasing=True)
            return Xv_t @ w
        return predict_fallback
    
    # Pick best
    candidates.sort(key=lambda t: t[1])
    best_pred, best_score = candidates[0]
    return best_pred

def fit(X_train, y_train):
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    x = X[:, 0]
    
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-8:
        x_std = 1.0
    
    predict = _fit_and_predict(x, y, x_mean, x_std)
    
    # Final guard: make sure predict returns finite values
    def safe_predict(X_test):
        X_t = np.asarray(X_test, dtype=float)
        try:
            out = predict(X_t)
            out = np.asarray(out, dtype=float).flatten()
            out = np.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
            return out
        except:
            # Ultimate fallback: constant
            return np.full(len(X_t), float(np.mean(y)))
    
    return safe_predict
