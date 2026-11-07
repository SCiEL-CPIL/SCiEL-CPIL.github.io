"""Symbolic regression: fast RBF + spline + polynomial ensemble with efficient CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 285.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _fit_rbf_fast(X, y, sigma, lam=1e-3):
    """Fast RBF with precomputed kernel. Returns predict."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2.0 * sigma**2))
    K_reg = K + lam * np.eye(n)

    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        try:
            w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
        except Exception:
            w = np.zeros(n)

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        diff_new = xn[:, None] - x[None, :]
        K_new = np.exp(-diff_new**2 / (2.0 * sigma**2))
        return K_new @ w

    return predict


def _fit_poly_fast(X, y, deg):
    """Polynomial fit with scaled input."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0

    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=deg + 1, increasing=True)

    try:
        w = np.linalg.lstsq(V, y, rcond=None)[0]
    except Exception:
        w = np.zeros(deg + 1)

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x_mean) / x_std
        Vn = np.vander(xn_scaled, N=deg + 1, increasing=True)
        return Vn @ w

    return predict


def _fit_spline_fast(X, y, s_factor):
    """Smoothing spline."""
    from scipy.interpolate import UnivariateSpline

    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(X)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    s = s_factor * np.var(y) * n

    try:
        sp = UnivariateSpline(x_sorted, y_sorted, s=s, k=4)

        def predict(X_new):
            return sp(X_new[:, 0].astype(np.float64))

        return predict
    except Exception:
        A = np.column_stack([np.ones(n), x])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            return predict
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            return predict


def _fit_fourier_fast(X, y, n_terms):
    """Fourier series fit."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    basis = np.ones((n, 1))
    for k in range(1, n_terms + 1):
        freq = 2.0 * np.pi * k / x_range
        basis = np.column_stack([basis, np.sin(freq * x), np.cos(freq * x)])

    try:
        w = np.linalg.lstsq(basis, y, rcond=None)[0]
    except Exception:
        w = np.zeros(basis.shape[1])

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        basis_new = np.ones((len(xn), 1))
        for k in range(1, n_terms + 1):
            freq = 2.0 * np.pi * k / x_range
            basis_new = np.column_stack([basis_new, np.sin(freq * xn), np.cos(freq * xn)])
        return basis_new @ w

    return predict


def _fit_chebyshev(X, y, deg):
    """Chebyshev polynomial fit."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0

    x_scaled = (2.0 * x - x_min - x_max) / (x_max - x_min)
    x_scaled = np.clip(x_scaled, -1.0, 1.0)

    # Build Chebyshev basis
    basis = np.ones((len(x), 1))
    if deg >= 1:
        T1 = x_scaled.copy()
        basis = np.column_stack([basis, T1])
    for k in range(2, deg + 1):
        T_k = 2.0 * x_scaled * basis[:, -1] - basis[:, -2]
        basis = np.column_stack([basis, T_k])

    try:
        w = np.linalg.lstsq(basis, y, rcond=None)[0]
    except Exception:
        w = np.zeros(basis.shape[1])

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (2.0 * xn - x_min - x_max) / (x_max - x_min)
        xn_scaled = np.clip(xn_scaled, -1.0, 1.0)

        basis_new = np.ones((len(xn), 1))
        if deg >= 1:
            T1 = xn_scaled.copy()
            basis_new = np.column_stack([basis_new, T1])
        for k in range(2, deg + 1):
            T_k = 2.0 * xn_scaled * basis_new[:, -1] - basis_new[:, -2]
            basis_new = np.column_stack([basis_new, T_k])

        return basis_new @ w

    return predict


def _kfold_cv(fit_func, X, y, k=5, seed=42):
    """K-fold cross-validation MSE."""
    n = len(X)
    if n < k + 2:
        k = max(2, n - 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    errs = []
    for f in folds:
        if _remaining() < 3:
            return np.inf
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[f], y[f]

        try:
            predict_func = fit_func(X_tr, y_tr)
            preds = predict_func(X_te)
            preds = np.asarray(preds, dtype=np.float64).flatten()
            if len(preds) != len(y_te):
                continue
            if not np.all(np.isfinite(preds)):
                continue
            err = np.mean((preds - y_te) ** 2)
            errs.append(err)
        except Exception:
            return np.inf

    if not errs:
        return np.inf
    return np.mean(errs)


def _ensemble_predict(predictors, weights):
    """Weighted ensemble of predictors."""
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        n = len(X_new)
        result = np.zeros(n, dtype=np.float64)
        for pred, w in zip(predictors, weights):
            try:
                r = pred(X_new)
                r = np.asarray(r, dtype=np.float64).flatten()
                if len(r) == n and np.all(np.isfinite(r)):
                    result += w * r
            except Exception:
                pass
        return result
    return predict


def fit(X_train, y_train):
    global _GLOBAL_START
    _GLOBAL_START = time.time()

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(X)

    if n < 4:
        c = np.mean(y) if n > 0 else 0.0

        def predict(X_new):
            return np.full(len(X_new), c)

        return predict

    x = X[:, 0]
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    # Build candidate list: (name, fit_func)
    candidates = []

    # RBF with various sigmas
    for sigma_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]:
        sigma = sigma_frac * x_range
        candidates.append((f"RBF({sigma_frac})", lambda X_tr, y_tr, s=sigma: _fit_rbf_fast(X_tr, y_tr, s)))

    # Polynomials
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        candidates.append((f"Poly({deg})", lambda X_tr, y_tr, d=deg: _fit_poly_fast(X_tr, y_tr, d)))

    # Chebyshev
    for deg in [3, 5, 8, 12, 16]:
        candidates.append((f"Cheb({deg})", lambda X_tr, y_tr, d=deg: _fit_chebyshev(X_tr, y_tr, d)))

    # Fourier
    for nt in [2, 3, 4, 5, 6, 8]:
        candidates.append((f"Fourier({nt})", lambda X_tr, y_tr, n=nt: _fit_fourier_fast(X_tr, y_tr, n)))

    # Splines
    for sf in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((f"Spline({sf})", lambda X_tr, y_tr, f=sf: _fit_spline_fast(X_tr, y_tr, f)))

    # Evaluate candidates
    results = []  # (cv_err, name, fit_func)
    for name, fit_func in candidates:
        if _remaining() < 5:
            break
        try:
            cv_err = _kfold_cv(fit_func, X, y, k=5)
            if np.isfinite(cv_err):
                results.append((cv_err, name, fit_func))
        except Exception:
            pass

    if not results:
        # Fallback: linear
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            return predict
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            return predict

    # Sort by CV error
    results.sort(key=lambda t: t[0])

    # Select top models for ensemble
    # Use top 3-5 models with weights inversely proportional to CV error
    top_k = min(5, len(results))
    top_models = results[:top_k]

    # If best model is much better, just use it
    best_err = top_models[0][0]
    if len(top_models) > 1 and top_models[1][0] > 2.0 * best_err:
        # Just use the best model
        best_model = top_models[0][2](X, y)
    else:
        # Ensemble of top models
        predictors = []
        weights = []
        for cv_err, name, fit_func in top_models:
            if _remaining() < 3:
                break
            try:
                pred = fit_func(X, y)
                predictors.append(pred)
                # Weight: inverse CV error, normalized
                weights.append(1.0 / (cv_err + 1e-10))
            except Exception:
                continue

        if not predictors:
            best_model = top_models[0][2](X, y)
        else:
            # Normalize weights
            w_sum = sum(weights)
            if w_sum > 0:
                weights = [w / w_sum for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)

            best_model = _ensemble_predict(predictors, weights)

    def safe_predict(X_new):
        try:
            r = best_model(np.asarray(X_new, dtype=np.float64))
            r = np.asarray(r, dtype=np.float64).flatten()
            r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
            return r
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
# EVOLVE-BLOCK-END
