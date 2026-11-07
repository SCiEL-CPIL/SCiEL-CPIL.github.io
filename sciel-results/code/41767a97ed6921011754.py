"""LOESS with adaptive bandwidth via LOOCV, plus quadratic fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _design_matrix(x, ddeg):
    """Build polynomial design matrix up to degree ddeg for points x (k, d)."""
    cols = [np.ones((len(x), 1))]
    if ddeg >= 1:
        cols.append(x)
    if ddeg >= 2:
        for i in range(x.shape[1]):
            for j in range(i, x.shape[1]):
                cols.append((x[:, i] * x[:, j])[:, np.newaxis])
    return np.hstack(cols)

def _loess_predict(X, y, X_pred, frac, deg):
    n = len(X)
    k = max(2, int(np.ceil(frac * n)))
    k = min(k, n)
    ddeg = min(deg, k - 1)
    diff = X_pred[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    order = np.argsort(dists, axis=1)
    idx_k = order[:, :k]
    d_k = np.take_along_axis(dists, idx_k, axis=1)
    d_max = np.maximum(d_k[:, -1:], 1e-12)
    ratio = d_k / d_max
    weights = np.where(ratio < 1.0, (1 - ratio ** 3) ** 3, 0.0)
    preds = np.zeros(len(X_pred))
    for i in range(len(X_pred)):
        x_loc = X[idx_k[i], :]
        w = weights[i]
        y_loc = y[idx_k[i]]
        A = _design_matrix(x_loc, ddeg)
        Aw = A * w[:, np.newaxis]
        try:
            beta = np.linalg.lstsq(Aw, y_loc * w, rcond=None)[0]
        except np.linalg.LinAlgError:
            Wsum = w.sum()
            preds[i] = np.sum(w * y_loc) / Wsum if Wsum > 1e-12 else y_loc.mean()
            continue
        xp = X_pred[i]
        cols = [np.array([1.0])]
        if ddeg >= 1:
            cols.append(xp.reshape(-1, 1))
        if ddeg >= 2:
            for i2 in range(xp.shape[0]):
                for j2 in range(i2, xp.shape[0]):
                    cols.append(np.array([xp[i2] * xp[j2]]))
        a_pred = np.hstack(cols).flatten()
        preds[i] = a_pred @ beta
    return preds

def _loess_loocv(X, y, frac, deg):
    n = len(X)
    k = max(2, int(np.ceil(frac * n)))
    k = min(k, n)
    ddeg = min(deg, k - 1)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mse_sum = 0.0
    for i in range(n):
        d_i = dists[i].copy()
        d_i[i] = np.inf
        order = np.argsort(d_i)
        idx = order[:k]
        d_k = d_i[idx]
        d_max = max(d_k[-1], 1e-12)
        ratio = d_k / d_max
        weights = np.where(ratio < 1.0, (1 - ratio ** 3) ** 3, 0.0)
        x_loc = X[idx, :]
        y_loc = y[idx]
        w = weights
        A = _design_matrix(x_loc, ddeg)
        Aw = A * w[:, np.newaxis]
        try:
            beta = np.linalg.lstsq(Aw, y_loc * w, rcond=None)[0]
        except np.linalg.LinAlgError:
            Wsum = w.sum()
            pred = np.sum(w * y_loc) / Wsum if Wsum > 1e-12 else y_loc.mean()
            mse_sum += (pred - y[i]) ** 2
            continue
        xp = X[i]
        cols = [np.array([1.0])]
        if ddeg >= 1:
            cols.append(xp.reshape(-1, 1))
        if ddeg >= 2:
            for i2 in range(xp.shape[0]):
                for j2 in range(i2, xp.shape[0]):
                    cols.append(np.array([xp[i2] * xp[j2]]))
        a_pred = np.hstack(cols).flatten()
        pred = a_pred @ beta
        mse_sum += (pred - y[i]) ** 2
    return mse_sum / n

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280.0
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n = len(X)
    fracs = [0.3, 0.5, 0.7, 0.9]
    deg = 1
    best_frac = None
    best_cv = np.inf
    best_pred_train = None
    for frac in fracs:
        if time.time() > deadline:
            break
        cv = _loess_loocv(X, y, frac, deg)
        if cv < best_cv:
            best_cv = cv
            best_frac = frac
            best_pred_train = _loess_predict(X, y, X, frac, deg)
    def features_quad(X):
        cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
        return np.column_stack(cols)
    try:
        w_quad, *_ = np.linalg.lstsq(features_quad(X), y, rcond=None)
        pred_quad_train = features_quad(X) @ w_quad
    except Exception:
        pred_quad_train = np.zeros_like(y)
    mse_loess = np.mean((best_pred_train - y) ** 2) if best_pred_train is not None else np.inf
    mse_quad = np.mean((pred_quad_train - y) ** 2)
    if best_frac is not None and mse_loess <= mse_quad * 1.1:
        frac = best_frac
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=float)
            return _loess_predict(X, y, X_pred, frac, deg)
    else:
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=float)
            return features_quad(X_pred) @ w_quad
    return predict
# EVOLVE-BLOCK-END
