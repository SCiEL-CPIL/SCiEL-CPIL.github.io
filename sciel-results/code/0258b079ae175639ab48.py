# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline

_TOTAL_DEADLINE = time.time() + 315.0  # safety margin under 330s


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


def _poly_features(X, degree):
    """Polynomial features up to given degree in d=1 (or general d)."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    # univariate powers
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    # cross terms
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for p in range(1, degree + 1):
                    for q in range(1, degree + 1 - p + 1):
                        if p + q <= degree:
                            cols.append((X[:, i] ** p) * (X[:, j] ** q))
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
        cols.append(np.sin(3 * x))
        cols.append(np.cos(3 * x))
        cols.append(np.tanh(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (x + 1e-9))
    return np.column_stack(cols)


def _mixed_features(X):
    """Combine polynomial, trig, and exp features."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(x)
        cols.append(x ** 2)
        cols.append(x ** 3)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
    return np.column_stack(cols)


def _fit_ols(F, y):
    """Solve least squares, return weights or None."""
    try:
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _rmse(F, w, y):
    pred = F @ w
    return np.sqrt(np.mean((pred - y) ** 2))


def _fit_spline(X, y, num_knots=10):
    """Fit a cubic spline via interpolation on sorted unique points."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        # Use a subset for knots to avoid issues
        n = len(xs)
        if n < num_knots + 4:
            num_knots = max(4, n - 4)
        knot_idx = np.linspace(0, n - 1, num_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ys[knot_idx]
        # Fit a spline through the knots
        s = make_interp_spline(knot_x, knot_y, k=3)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_mlp(X, y, hidden_sizes=(16, 16), lr=0.01, epochs=200):
    """Small MLP via simple gradient descent."""
    try:
        d = X.shape[1]
        n = X.shape[0]
        # Normalize inputs
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma

        np.random.seed(42)
        layers = [d] + list(hidden_sizes) + [1]
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
        b = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]

        def forward(Xb):
            a = Xb
            acts = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                acts.append(a)
            return acts

        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            acts = forward(Xn)
            pred = acts[-1]
            # Backward
            dz = (pred - yn) / n
            grads_W = [None] * len(W)
            grads_b = [None] * len(b)
            for i in range(len(W) - 1, -1, -1):
                if i == len(W) - 1:
                    dz_i = dz
                else:
                    dz_i = dz @ W[i].T * (1 - acts[i+1]**2)
                grads_W[i] = acts[i].T @ dz_i
                grads_b[i] = dz_i.sum(axis=0)
            for i in range(len(W)):
                W[i] -= lr * grads_W[i]
                b[i] -= lr * grads_b[i]

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            a = Xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        return predict
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y):
    """Try fitting a linear model on given features. Returns (predict_fn, rmse) or (None, inf)."""
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y)
        if w is None:
            return None, np.inf
        rmse = _rmse(F, w, y)
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try polynomial degrees
    for deg in [1, 2, 3, 4, 5, 6]:
        if _remaining() < 10.0:
            break
        predict, rmse = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try trig
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_trig_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try exp/log
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_exp_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try mixed
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_mixed_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try spline
    if _remaining() > 10.0 and d == 1:
        sp_predict = _fit_spline(X_train, y_train)
        if sp_predict is not None:
            # Evaluate rmse on training data
            pred = sp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = sp_predict

    # Try MLP
    if _remaining() > 30.0:
        mlp_predict = _fit_mlp(X_train, y_train)
        if mlp_predict is not None:
            pred = mlp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = mlp_predict

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
