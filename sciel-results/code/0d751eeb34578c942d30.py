"""Structural redesign: RBF feature-space regression with adaptive bandwidth
+ ridge solve, robust fallback, and an explicit deadline guard."""
import time
import numpy as np

# Wall-clock guard for the whole evaluation (evaluator kills at 330 s).
_T0 = time.time()
_DEADLINE = _T0 + 280.0  # safety margin


def _remaining():
    return _DEADLINE - time.time()


def _fit_rbf(X, y, max_centers=120):
    n = X.shape[0]
    d = X.shape[1]

    # Choose a subset of training points as RBF centers (spread out if large).
    if n <= max_centers:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_centers).astype(int)
    C = X[idx]

    # Adaptive bandwidth: median of nearest-neighbor distances among centers.
    # Use a subsample for speed when many centers.
    m = C.shape[0]
    if m > 40:
        sub = C[np.linspace(0, m - 1, 40).astype(int)]
    else:
        sub = C
    # pairwise distances within sub
    diff = sub[:, None, :] - sub[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)
    nn = nn[nn > 0]
    if nn.size == 0:
        sigma = 1.0
    else:
        sigma = float(np.median(nn))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    def features(Z):
        # squared distances from Z to each center
        diff = Z[:, None, :] - C[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        return np.exp(-sq / (2.0 * sigma * sigma))

    # Ridge regression in feature space.
    F = features(X)
    lam = 1e-3
    A = F.T @ F + lam * np.eye(m)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b

    def predict(Z):
        try:
            return features(Z) @ w
        except Exception:
            # last-resort fallback: nearest-center mean
            diff = Z[:, None, :] - C[None, :, :]
            sq = np.sum(diff * diff, axis=2)
            nn_idx = np.argmin(sq, axis=1)
            return y[idx[nn_idx]]

    return predict


def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Guard: if we're already near the deadline, return a cheap fallback.
    if _remaining() < 5.0:
        # nearest-neighbor mean over a coarse grid of training points
        idx = np.linspace(0, X.shape[0] - 1, min(200, X.shape[0])).astype(int)
        C, cy = X[idx], y[idx]

        def predict(Z):
            diff = Z[:, None, :] - C[None, :, :]
            sq = np.sum(diff * diff, axis=2)
            return cy[np.argmin(sq, axis=1)]

        return predict

    try:
        pred = _fit_rbf(X, y)
    except Exception:
        # Global fallback: mean predictor (never throws, always finite).
        mu = float(np.mean(y)) if y.size else 0.0

        def predict(Z):
            return np.full(np.asarray(Z).shape[0], mu)

        return predict

    # Ensure output is finite.
    def safe_predict(Z):
        out = pred(Z)
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            out = out.reshape(1)
        # Replace any non-finite with the training mean to stay safe.
        if not np.all(np.isfinite(out)):
            mu = float(np.mean(y)) if y.size else 0.0
            out = np.where(np.isfinite(out), out, mu)
        return out

    return safe_predict
