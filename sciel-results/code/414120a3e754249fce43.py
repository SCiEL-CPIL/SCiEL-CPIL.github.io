"""
Structural change: use PCHIP (monotone cubic) interpolation as the primary model for
1-D noiseless data, with a fallback to a rich polynomial/spline ensemble selected via
cross-validation.  The key insight is that 200 noiseless 1-D samples can be
interpolated near-perfectly if the interpolant respects local monotonicity (PCHIP),
avoiding Runge oscillations of high-degree polynomials.

We also add a Gaussian Process with a Matern-5/2 kernel as an alternative smooth
model, and select the best per-equation model via k-fold CV on the training set.

A hard deadline guard ensures we return the best model found before the 330 s wall-clock
limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# Global deadline: 320 s (10 s safety margin before 330 s kill)
_DEADLINE = time.time() + 320.0


def _elapsed_ok():
    return time.time() < _DEADLINE


def _pchip_predict(x_train, y_train, X):
    """PCHIP interpolation on sorted 1-D data."""
    x = x_train.ravel()
    y = y_train.ravel()
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Remove duplicate x values (keep last)
    mask = np.concatenate(([True], np.diff(x_sorted) > 1e-12))
    x_u = x_sorted[mask]
    y_u = y_sorted[mask]
    pchip = PchipInterpolator(x_u, y_u)
    return pchip(X.ravel()).reshape(X.shape)


def _spline_predict(x_train, y_train, X, s=0.0):
    """Univariate spline with s=0 (interpolating) or small s."""
    x = x_train.ravel()
    y = y_train.ravel()
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    mask = np.concatenate(([True], np.diff(x_sorted) > 1e-12))
    x_u = x_sorted[mask]
    y_u = y_sorted[mask]
    if len(x_u) < 5:
        return np.full(X.shape, np.mean(y_u))
    try:
        spl = UnivariateSpline(x_u, y_u, s=s, k=3)
        return spl(X.ravel()).reshape(X.shape)
    except Exception:
        return _pchip_predict(x_train, y_train, X)


def _poly_predict(x_train, y_train, X, degree=8):
    """Polynomial regression via least squares."""
    x = x_train.ravel()
    y = y_train.ravel()
    V = np.vander(x, degree + 1, increasing=True)
    w, *_ = np.linalg.lstsq(V, y, rcond=None)
    V_test = np.vander(X.ravel(), degree + 1, increasing=True)
    return (V_test @ w).reshape(X.shape)


def _gp_predict(x_train, y_train, X, length_scale=0.5, sigma=1.0):
    """Gaussian Process with Matern-5/2 kernel, noise-free."""
    x = x_train.ravel()
    y = y_train.ravel()
    Xq = X.ravel()

    def matern52(a, b):
        d = np.abs(a - b)
        r = d * length_scale
        return sigma ** 2 * (1 + r + r * r / 3.0) * np.exp(-r)

    K = matern52(x, x) + 1e-12 * np.eye(len(x))
    Ks = matern52(x, Xq)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        pred = Ks.T @ alpha
        return pred.reshape(X.shape)
    except np.linalg.LinAlgError:
        try:
            K += 1e-6 * np.eye(len(x))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            pred = Ks.T @ alpha
            return pred.reshape(X.shape)
        except Exception:
            return _pchip_predict(x_train, y_train, X)


def _cv_score(model_fn, x_train, y_train, n_folds=5):
    """K-fold CV MSE on 1-D data."""
    n = len(x_train)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    total_mse = 0.0
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        trn_idx = np.concatenate([indices[:start], indices[end:]])
        x_tr = x_train[trn_idx]
        y_tr = y_train[trn_idx]
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        try:
            pred = model_fn(x_tr, y_tr, x_val)
            mse = np.mean((pred - y_val) ** 2)
            total_mse += mse
        except Exception:
            total_mse += np.inf
    return total_mse / n_folds


def fit(X_train, y_train):
    t_start = time.time()
    X = X_train
    y = y_train

    # All equations are 1-D per the task contract
    d = X.shape[1]
    if d == 1:
        x_train = X.ravel()
        y_train_1d = y.ravel()
    else:
        # Fallback for multi-dim: use polynomial
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        return lambda X: features(X) @ w

    # Candidate models with their hyperparameters
    candidates = []

    # PCHIP (primary)
    candidates.append(("pchip", lambda xt, yt, Xq: _pchip_predict(xt, yt, Xq)))

    # Univariate spline, interpolating
    candidates.append(("spline0", lambda xt, yt, Xq: _spline_predict(xt, yt, Xq, s=0.0)))

    # Univariate spline, small smoothing
    candidates.append(("spline_s", lambda xt, yt, Xq: _spline_predict(xt, yt, Xq, s=1e-3)))

    # Polynomial degrees
    for deg in [4, 6, 8]:
        candidates.append((f"poly{deg}", lambda xt, yt, Xq, d=deg: _poly_predict(xt, yt, Xq, d)))

    # GP with different length scales
    for ls in [0.3, 0.5, 1.0]:
        candidates.append((f"gp_{ls}", lambda xt, yt, Xq, l=ls: _gp_predict(xt, yt, Xq, length_scale=l)))

    # Evaluate candidates via CV, but respect deadline
    best_model = None
    best_cv = np.inf
    best_name = "pchip"

    # Always keep PCHIP as a safe default
    default_pred = _pchip_predict(x_train, y_train, X_train)

    for name, fn in candidates:
        if not _elapsed_ok():
            break
        try:
            cv = _cv_score(fn, x_train, y_train, n_folds=5)
            if cv < best_cv:
                best_cv = cv
                best_model = fn
                best_name = name
        except Exception:
            pass

    # If deadline is close, just use PCHIP
    if time.time() > _DEADLINE - 30:
        best_model = lambda xt, yt, Xq: _pchip_predict(xt, yt, Xq)
        best_name = "pchip_fallback"

    # Final prediction function
    def predict(X):
        if d == 1:
            try:
                return best_model(x_train, y_train, X)
            except Exception:
                return _pchip_predict(x_train, y_train, X)
        else:
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            F = np.column_stack(cols)
            return F @ w

    return predict
# EVOLVE-BLOCK-END
