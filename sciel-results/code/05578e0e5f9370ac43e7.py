"""Symbolic regression: multi-basis ensemble with adaptive selection via CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

_DEADLINE = 300.0  # safety margin before 330s kill
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _remaining():
    return _DEADLINE - _elapsed()


def _poly_features(X, degree=3):
    """Polynomial features up to given degree (all interactions)."""
    d = X.shape[1]
    cols = [np.ones((X.shape[0], 1))]
    for deg in range(1, degree + 1):
        for idx in product(range(d), repeat=deg):
            # only include if sorted to avoid duplicates
            if idx != tuple(sorted(idx)):
                continue
            feat = np.ones(X.shape[0])
            for i in idx:
                feat = feat * X[:, i]
            cols.append(feat)
    return np.column_stack(cols)


def _rbf_features(X, centers, bandwidths):
    """RBF features with given centers and bandwidths."""
    n = X.shape[0]
    d = X.shape[1]
    cols = []
    for c in centers:
        for bw in bandwidths:
            diff = X - c
            dist_sq = np.sum(diff ** 2, axis=1)
            cols.append(np.exp(-dist_sq / (2.0 * bw ** 2)))
    return np.column_stack(cols) if cols else np.ones((n, 1))


def _fourier_features(X, freqs):
    """Fourier (sin/cos) features."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones((n, 1))]
    for j in range(d):
        for f in freqs:
            cols.append(np.sin(2 * np.pi * f * X[:, j]))
            cols.append(np.cos(2 * np.pi * f * X[:, j]))
    return np.column_stack(cols)


def _fit_ridge(X, y, lam):
    """Fit ridge regression. Returns (w, intercept)."""
    n, p = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb
    A[np.diag_indices(p + 1)] += lam
    b = Xb.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Xb, y, rcond=None)[0]
    return w


def _ridge_predict(X, w):
    """Predict with ridge coefficients (last is intercept)."""
    p = X.shape[1]
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w


def _cross_val_score(X, y, feature_fn, n_folds=5, lam=1.0):
    """Compute mean CV MSE for a feature function."""
    n = X.shape[0]
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    mses = []
    for k in range(n_folds):
        val_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        F_tr = feature_fn(X_tr)
        F_val = feature_fn(X_val)
        if F_tr.shape[1] == 0:
            F_tr = np.ones((F_tr.shape[0], 1))
            F_val = np.ones((F_val.shape[0], 1))
        w = _fit_ridge(F_tr, y_tr, lam)
        pred = _ridge_predict(F_val, w)
        mse = np.mean((pred - y_val) ** 2)
        mses.append(mse)
    return np.mean(mses)


def _safe_predict(X, models):
    """Average predictions from a list of (feature_fn, w) tuples."""
    if not models:
        return np.zeros(X.shape[0])
    preds = []
    for feature_fn, w in models:
        try:
            F = feature_fn(X)
            if F.shape[1] == 0:
                F = np.ones((X.shape[0], 1))
            p = _ridge_predict(F, w)
            p = np.asarray(p, dtype=float)
            if p.shape[0] != X.shape[0]:
                raise ValueError("shape mismatch")
            if not np.all(np.isfinite(p)):
                raise ValueError("non-finite")
            preds.append(p)
        except Exception:
            continue
    if not preds:
        return np.zeros(X.shape[0])
    return np.mean(preds, axis=0)


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    n, d = X_train.shape

    # Normalize inputs to [0, 1] range for stable features
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0

    def _norm(X):
        return (X - x_min) / x_range

    Xn = _norm(X_train)

    # Build candidate feature functions
    candidates = []

    # Polynomial degrees 1, 2, 3
    for deg in [1, 2, 3]:
        candidates.append(('poly', deg, lambda X, _d=deg: _poly_features(_norm(X), _d)))

    # RBF with various bandwidths and centers
    centers = [np.zeros(d)]
    if d >= 1:
        centers.append(Xn.mean(axis=0))
        centers.append(x_min / x_range)  # 0 in normalized
        centers.append(np.ones(d))       # 1 in normalized
    bandwidths = [0.1, 0.3, 0.5, 1.0, 2.0]
    for c in centers:
        for bw in bandwidths:
            candidates.append(('rbf', (c, bw), lambda X, _c=c, _b=bw: _rbf_features(_norm(X), [_c], [_b])))

    # Fourier with various frequencies
    freqs_options = [
        [0.5, 1.0],
        [1.0, 2.0],
        [0.5, 1.0, 2.0],
        [1.0, 1.5, 2.0],
    ]
    for f in freqs_options:
        candidates.append(('fourier', f, lambda X, _f=f: _fourier_features(_norm(X), _f)))

    # Evaluate candidates via CV
    best_models = []
    best_scores = []
    lam_options = [0.01, 0.1, 1.0, 10.0]

    for name, key, feature_fn in candidates:
        if _remaining() < 10.0:
            break
        best_cv = np.inf
        best_lam = 1.0
        for lam in lam_options:
            if _remaining() < 5.0:
                break
            try:
                cv = _cross_val_score(Xn, y_train, feature_fn, n_folds=5, lam=lam)
                if cv < best_cv:
                    best_cv = cv
                    best_lam = lam
            except Exception:
                continue
        if best_cv < np.inf:
            best_scores.append(best_cv)
            best_models.append((name, key, feature_fn, best_lam, best_cv))

    # Sort by CV score, take top K
    best_models.sort(key=lambda x: x[4])
    K = min(5, len(best_models))

    # Fit each top model on full training data
    final_models = []
    for name, key, feature_fn, lam, cv in best_models[:K]:
        if _remaining() < 3.0:
            break
        try:
            F = feature_fn(Xn)
            if F.shape[1] == 0:
                F = np.ones((F.shape[0], 1))
            w = _fit_ridge(F, y_train, lam)
            final_models.append((feature_fn, w))
        except Exception:
            continue

    # Fallback: simple linear model if nothing worked
    if not final_models:
        def lin_feat(X):
            return _norm(X)
        try:
            F = lin_feat(Xn)
            w = _fit_ridge(F, y_train, 1.0)
            final_models.append((lin_feat, w))
        except Exception:
            # Ultimate fallback: mean
            mean_y = np.mean(y_train)
            def _const_predict(X):
                return np.full(X.shape[0], mean_y)
            return _const_predict

    # Ensemble predict
    def predict(X):
        X = np.asarray(X, dtype=float)
        out = _safe_predict(X, final_models)
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    return predict
# EVOLVE-BLOCK-END
