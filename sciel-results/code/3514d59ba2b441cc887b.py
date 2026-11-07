"""Symbolic regression via multi-basis least-squares with time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator
import math

# Global deadline guard (330s total, use 320s safety margin)
_DEADLINE = time.time() + 315.0

def _time_left():
    return _DEADLINE - time.time()


def _make_rbf_predictor(X_train, y_train, n_neighbors=10):
    """KNN-based predictor using local linear regression."""
    from scipy.spatial import cKDTree
    
    tree = cKDTree(X_train)
    
    def predict(X):
        X = np.atleast_2d(X)
        n = len(X)
        preds = np.zeros(n)
        for i in range(n):
            dists, idxs = tree.query(X[i], k=min(n_neighbors, len(X_train)))
            if isinstance(idxs, (int, np.integer)):
                idxs = np.array([idxs])
            # Local linear regression
            X_local = X_train[idxs]
            y_local = y_train[idxs]
            # Add bias
            X_local_b = np.column_stack([X_local, np.ones(len(X_local))])
            x_new = np.column_stack([X[i], [1.0]])
            try:
                w, *_ = np.linalg.lstsq(X_local_b, y_local, rcond=None)
                preds[i] = x_new @ w
            except:
                preds[i] = np.mean(y_local)
        return preds
    
    return predict


def _make_rbf_predictor_v2(X_train, y_train):
    """RBF interpolation with smoothing."""
    # Use RBF with Gaussian kernel
    rbf = RBFInterpolator(X_train, y_train, kernel='gaussian', smoothing=0.1)
    return lambda X: rbf(np.atleast_2d(X))


def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
    # Quadratic and higher
    for deg in range(2, degree + 1):
        for i in range(d):
            for j in range(i, d):
                if deg == 2:
                    if i == j:
                        cols.append(X[:, i] ** 2)
                    else:
                        cols.append(X[:, i] * X[:, j])
                else:
                    # For degree > 2, include pure powers and cross terms
                    # Simplified: just pure powers for each variable
                    if i == j:
                        cols.append(X[:, i] ** deg)
    return np.column_stack(cols)


def _trig_features(X):
    """Generate trigonometric features."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
    return np.column_stack(cols)


def _exp_log_features(X):
    """Generate exponential and log features."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.log(x + 1e-8))
        cols.append(1.0 / (x + 1e-8))
    return np.column_stack(cols)


def _power_features(X):
    """Generate power features."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
            cols.append(np.abs(x) ** p)
    return np.column_stack(cols)


def _combined_features(X, mode):
    """Combine different feature sets."""
    if mode == 'poly3':
        return _poly_features(X, 3)
    elif mode == 'poly2':
        return _poly_features(X, 2)
    elif mode == 'trig':
        return _trig_features(X)
    elif mode == 'exp_log':
        return _exp_log_features(X)
    elif mode == 'power':
        return _power_features(X)
    elif mode == 'all':
        # Combine poly2 + trig + exp_log
        f1 = _poly_features(X, 2)
        f2 = _trig_features(X)
        f3 = _exp_log_features(X)
        f4 = _power_features(X)
        return np.column_stack([f1, f2, f3, f4])
    elif mode == 'poly2_trig':
        f1 = _poly_features(X, 2)
        f2 = _trig_features(X)
        return np.column_stack([f1, f2])
    elif mode == 'poly2_exp':
        f1 = _poly_features(X, 2)
        f2 = _exp_log_features(X)
        return np.column_stack([f1, f2])
    else:
        return _poly_features(X, 2)


def _fit_linear(features_func, X_train, y_train, mode):
    """Fit linear model with given feature function."""
    try:
        F_train = features_func(X_train, mode)
        # Ridge regularization for stability
        alpha = 1e-6
        A = F_train.T @ F_train + alpha * np.eye(F_train.shape[1])
        b = F_train.T @ y_train
        w = np.linalg.solve(A, b)
        
        def predict(X):
            X = np.atleast_2d(X)
            F = features_func(X, mode)
            return F @ w
        return predict
    except Exception:
        return None


def _fit_spline_1d(x_train, y_train):
    """Fit 1D spline for single-input problems."""
    from scipy.interpolate import UnivariateSpline
    try:
        order = 3
        s = 0  # No smoothing, exact interpolation
        spl = UnivariateSpline(x_train, y_train, k=order, s=s)
        return lambda X: spl(np.atleast_2d(X)[:, 0])
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.atleast_2d(X_train)
    y_train = np.asarray(y_train, dtype=float)
    d = X_train.shape[1]
    
    best_predict = None
    best_score = np.inf
    
    # Try multiple feature spaces and pick the one with best training MSE
    # (since data is noiseless, best training fit should generalize well)
    
    modes_to_try = []
    
    if d == 1:
        modes_to_try = ['poly3', 'poly2_trig', 'trig', 'exp_log', 'power', 'all', 'poly2']
    elif d == 2:
        modes_to_try = ['poly3', 'poly2_trig', 'poly2_exp', 'all', 'poly2']
    else:
        modes_to_try = ['poly3', 'poly2', 'poly2_trig']
    
    # For 1D, also try spline
    spline_predict = None
    if d == 1 and _time_left() > 5:
        spline_predict = _fit_spline_1d(X_train[:, 0], y_train)
        if spline_predict is not None:
            try:
                pred = spline_predict(X_train)
                mse = np.mean((pred - y_train) ** 2)
                var_y = np.var(y_train)
                if var_y > 1e-12:
                    score = mse / var_y
                    if score < best_score:
                        best_score = score
                        best_predict = spline_predict
            except Exception:
                pass
    
    # Try various feature spaces
    for mode in modes_to_try:
        if _time_left() < 2:
            break
        try:
            predict = _fit_linear(_combined_features, X_train, y_train, mode)
            if predict is not None:
                pred = predict(X_train)
                mse = np.mean((pred - y_train) ** 2)
                var_y = np.var(y_train)
                if var_y > 1e-12:
                    score = mse / var_y
                    if score < best_score:
                        best_score = score
                        best_predict = predict
        except Exception:
            continue
    
    # If no good fit found, try KNN as fallback
    if best_predict is None or best_score > 1e-6:
        if _time_left() > 3:
            try:
                knn_predict = _make_rbf_predictor(X_train, y_train, n_neighbors=5)
                pred = knn_predict(X_train)
                mse = np.mean((pred - y_train) ** 2)
                var_y = np.var(y_train)
                if var_y > 1e-12:
                    score = mse / var_y
                    if score < best_score:
                        best_score = score
                        best_predict = knn_predict
            except Exception:
                pass
    
    # Final fallback: simple mean predictor
    if best_predict is None:
        y_mean = np.mean(y_train)
        best_predict = lambda X: np.full(len(np.atleast_2d(X)), y_mean)
    
    # Ensure predictions are finite
    def safe_predict(X):
        X = np.atleast_2d(X)
        try:
            result = best_predict(X)
            result = np.asarray(result, dtype=float)
            # Replace non-finite values
            if not np.all(np.isfinite(result)):
                result = np.where(np.isfinite(result), result, np.mean(y_train))
            return result
        except Exception:
            return np.full(len(X), np.mean(y_train))
    
    return safe_predict
# EVOLVE-BLOCK-END
