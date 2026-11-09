"""Combined symbolic regression: polynomial basis + nonlinear features + GP fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill

    d = X_train.shape[1]
    n = len(X_train)
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    best_score = np.inf
    best_predictor = None

    def _norm_mse(pred, y):
        return np.mean((pred - y) ** 2) / y_var

    def _try_update(model_fn):
        nonlocal best_score, best_predictor
        try:
            pred = model_fn(X_train)
            if not np.all(np.isfinite(pred)):
                return
            sc = _norm_mse(pred, y_train)
            if sc < best_score:
                best_score = sc
                best_predictor = model_fn
        except Exception:
            pass

    # --- Strategy 1: Polynomial regression with increasing degree ---
    def make_poly_features(X, degree):
        cols = [np.ones(X.shape[0])]
        for total_deg in range(1, degree + 1):
            for indices in product(range(d), repeat=total_deg):
                if sorted(indices) != indices:
                    continue
                feat = np.ones(X.shape[0])
                for idx in indices:
                    feat = feat * X[:, idx]
                cols.append(feat)
        return np.column_stack(cols)

    for deg in [2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
        try:
            F = make_poly_features(X_train, deg)
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            _try_update(lambda X, F_fn=lambda X: make_poly_features(X, deg), w=w: F_fn(X) @ w)
        except Exception:
            pass

    # --- Strategy 2: Nonlinear basis features + ridge regression ---
    if time.time() < deadline - 20:
        base_funcs = [
            lambda x: x,
            lambda x: x ** 2,
            lambda x: x ** 3,
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.exp(-x ** 2 / 10.0),
            lambda x: np.log1p(x),
            lambda x: 1.0 / (1.0 + np.exp(-x)),
            lambda x: x * np.sin(x),
            lambda x: x * np.cos(x),
        ]

        candidate_features = [np.ones(n)]

        # Single-variable transforms
        for j in range(d):
            for f in base_funcs:
                try:
                    feat = f(X_train[:, j])
                    if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                        candidate_features.append(feat)
                except Exception:
                    pass

        # Pairwise products (limited for speed)
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    for f1 in base_funcs[:5]:
                        for f2 in base_funcs[:5]:
                            try:
                                feat = f1(X_train[:, i]) * f2(X_train[:, j])
                                if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                                    candidate_features.append(feat)
                            except Exception:
                                pass

        # Pairwise sums
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    for f1 in base_funcs[:3]:
                        for f2 in base_funcs[:3]:
                            try:
                                feat = f1(X_train[:, i]) + f2(X_train[:, j])
                                if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                                    candidate_features.append(feat)
                            except Exception:
                                pass

        F_cand = np.column_stack(candidate_features)
        n_feat = F_cand.shape[1]

        for alpha in [0, 0.01, 0.1, 1.0, 10.0]:
            if time.time() > deadline - 5:
                break
            try:
                if alpha == 0:
                    w, *_ = np.linalg.lstsq(F_cand, y_train, rcond=None)
                else:
                    A = F_cand.T @ F_cand + alpha * np.eye(n_feat)
                    b = F_cand.T @ y_train
                    w = np.linalg.solve(A, b)
                _try_update(lambda X, F=F_cand, w=w: F @ w)
            except Exception:
                pass

    # --- Strategy 3: GP regression fallback ---
    if time.time() < deadline - 15:
        try:
            from scipy.spatial.distance import cdist

            def gp_predict(X_pred, X_tr, y_tr, length_scale, sigma_f, noise):
                K_tr = sigma_f ** 2 * np.exp(-cdist(X_tr, X_tr) ** 2 / (2 * length_scale ** 2)) + noise * np.eye(len(X_tr))
                K_pred = sigma_f ** 2 * np.exp(-cdist(X_pred, X_tr) ** 2 / (2 * length_scale ** 2))
                L = np.linalg.cholesky(K_tr)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr - y_mean))
                mu = y_mean + K_pred @ alpha
                return mu

            for ls in [0.5, 1.0, 2.0, 5.0]:
                for sf in [1.0, 5.0, 10.0]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        _try_update(lambda X, X_t=X_train, y_t=y_train, l=ls, s=sf: gp_predict(X, X_t, y_t, l, s, 1e-8))
                    except Exception:
                        pass
        except Exception:
            pass

    # Final fallback: constant predictor
    if best_predictor is None:
        best_predictor = lambda X: np.full(X.shape[0], y_mean)

    return best_predictor
# EVOLVE-BLOCK-END
