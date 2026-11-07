"""Symbolic regression via kernel ridge + basis portfolio with LOOCV selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 330.0
_SAFETY = 20.0


def _time_left():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _SAFETY
    return _DEADLINE - time.time()


def _loocv_ridge(B, y, lam):
    """Efficient LOOCV MSE for ridge regression using the hat-matrix formula."""
    n, p = B.shape
    A = B.T @ B + lam * np.eye(p)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    B_Ainv = B @ A_inv
    h = np.einsum('ij,ij->i', B, B_Ainv)
    yhat = B @ (A_inv @ (B.T @ y))
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    loocv = np.mean((resid / denom) ** 2)
    return loocv


def _loocv_kernel(K, y, lam):
    """Efficient LOOCV MSE for kernel ridge regression."""
    n = K.shape[0]
    A = K + lam * np.eye(n)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    K_Ainv = K @ A_inv
    h = np.einsum('ij,ij->i', K, K_Ainv)
    yhat = K @ (A_inv @ y)
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    loocv = np.mean((resid / denom) ** 2)
    return loocv


def _rbf_kernel(X1, X2, sigma):
    """Compute RBF kernel matrix between X1 and X2."""
    X1 = X1[:, 0:1]
    X2 = X2[:, 0:1]
    d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-d2 / (2 * sigma**2))


def _poly_kernel(X1, X2, degree, gamma=1.0):
    """Compute polynomial kernel matrix."""
    X1 = X1[:, 0:1]
    X2 = X2[:, 0:1]
    return (gamma * X1 @ X2.T + 1.0) ** degree


def _chebyshev_terms(x, max_deg):
    """Return list of Chebyshev polynomials T_0..T_max_deg evaluated at x."""
    n = x.shape[0]
    t0 = np.ones(n)
    t1 = x.copy()
    out = [t0, t1]
    for k in range(2, max_deg + 1):
        tk = 2.0 * x * t1 - t0
        out.append(tk)
        t0, t1 = t1, tk
    return out


def _build_basis(x, max_poly, n_fourier):
    """Build a rich basis for 1-D input x."""
    n = x.shape[0]
    cols = [np.ones(n)]
    for t in _chebyshev_terms(x, max_poly):
        cols.append(t)
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _build_exp_basis(x):
    """Build an exponential/logarithmic basis for 1-D input x."""
    n = x.shape[0]
    cols = [np.ones(n), x]
    cols.append(np.exp(x))
    cols.append(np.exp(-x))
    pos_mask = x > 0
    safe_x = np.where(pos_mask, x, 1.0)
    cols.append(np.log(safe_x))
    cols.append(np.log1p(np.abs(x)))
    return np.column_stack(cols)


def _build_rational_basis(x):
    """Build a rational basis for 1-D input x."""
    n = x.shape[0]
    safe_x = np.where(np.abs(x) > 1e-6, x, 1e-6)
    cols = [np.ones(n), x, x**2, x**3, 1.0/safe_x, x/safe_x**2]
    return np.column_stack(cols)


def _build_tanh_basis(x):
    """Build a tanh/sigmoid basis for 1-D input x."""
    n = x.shape[0]
    cols = [np.ones(n), x]
    cols.append(np.tanh(x))
    cols.append(np.tanh(2 * x))
    cols.append(np.tanh(0.5 * x))
    cols.append(1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))
    return np.column_stack(cols)


def _build_hinge_basis(x, thresholds):
    """Build a hinge (ReLU-like) basis for 1-D input x."""
    n = x.shape[0]
    cols = [np.ones(n)]
    for t in thresholds:
        cols.append(np.maximum(0, x - t))
        cols.append(np.maximum(0, t - x))
    return np.column_stack(cols)


def fit(X_train, y_train):
    global _DEADLINE
    x = X_train[:, 0].astype(np.float64)
    y = y_train.astype(np.float64)
    n = x.shape[0]

    # Determine input range
    lo, hi = float(np.min(x)), float(np.max(x))
    span = max(hi - lo, 1e-6)

    best_cv = np.inf
    best_predict = None

    t_start = time.time()
    time_budget = 280.0  # Leave 50s for safety

    # --- Model 1: Spline (Interpolating) ---
    # For noiseless data, exact interpolation is often ideal.
    # We use 5-Fold CV to estimate generalization.
    def fit_spline(x_tr, y_tr):
        from scipy.interpolate import UnivariateSpline
        # s=0 for exact interpolation
        try:
            return UnivariateSpline(x_tr, y_tr, s=0, k=3)
        except Exception:
            return None

    def predict_spline(model, x_val):
        if model is None:
            return np.zeros_like(x_val)
        return model(x_val)

    # 5-Fold CV for spline
    def kfold_cv_spline():
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        scores = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            try:
                model = fit_spline(x[train_idx], y[train_idx])
                pred = predict_spline(model, x[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)
            except Exception:
                return np.inf
        return np.mean(scores)

    cv_spline = kfold_cv_spline()
    if cv_spline < best_cv:
        best_cv = cv_spline
        final_spline = fit_spline(x, y)
        if final_spline is not None:
            def predict_final(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                out = final_spline(X[:, 0])
                return np.where(np.isfinite(out), out, 0.0)
            best_predict = predict_final

    # --- Model 2: Kernel Ridge with RBF ---
    sigma_candidates = [span * s for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]
    lam_candidates = np.logspace(-6, 4, 15)

    for sigma in sigma_candidates:
        if time.time() - t_start > time_budget:
            break
        K = _rbf_kernel(X_train, X_train, sigma)
        for lam in lam_candidates:
            if time.time() - t_start > time_budget:
                break
            cv = _loocv_kernel(K, y, lam)
            if cv < best_cv:
                best_cv = cv
                A = K + lam * np.eye(n)
                try:
                    alpha = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    continue

                def predict_rbf(X, K_train=K, alpha=alpha, sigma=sigma, x_train=x):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    K_test = _rbf_kernel(X, X_train, sigma)
                    out = K_test @ alpha
                    return np.where(np.isfinite(out), out, 0.0)

                best_predict = predict_rbf

    # --- Model 3: Kernel Ridge with Polynomial ---
    for degree in [2, 3, 4, 5]:
        if time.time() - t_start > time_budget:
            break
        gamma_candidates = [1.0, 0.5, 0.25, 0.1]
        for gamma in gamma_candidates:
            if time.time() - t_start > time_budget:
                break
            K = _poly_kernel(X_train, X_train, degree, gamma)
            for lam in lam_candidates:
                if time.time() - t_start > time_budget:
                    break
                cv = _loocv_kernel(K, y, lam)
                if cv < best_cv:
                    best_cv = cv
                    A = K + lam * np.eye(n)
                    try:
                        alpha = np.linalg.solve(A, y)
                    except np.linalg.LinAlgError:
                        continue

                    def predict_poly(X, K_train=K, alpha=alpha, degree=degree, gamma=gamma, x_train=x):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(1, -1)
                        K_test = _poly_kernel(X, X_train, degree, gamma)
                        out = K_test @ alpha
                        return np.where(np.isfinite(out), out, 0.0)

                    best_predict = predict_poly

    # --- Model 4: Basis Ridge (Chebyshev + Fourier) ---
    max_poly = 20
    n_fourier = 10
    B = _build_basis(x, max_poly, n_fourier)
    p = B.shape[1]
    for lam in lam_candidates:
        if time.time() - t_start > time_budget:
            break
        cv = _loocv_ridge(B, y, lam)
        if cv < best_cv:
            best_cv = cv
            A = B.T @ B + lam * np.eye(p)
            try:
                w = np.linalg.solve(A, B.T @ y)
            except np.linalg.LinAlgError:
                continue

            def predict_basis(X, B_train=B, w=w, max_poly=max_poly, n_fourier=n_fourier):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                Bx = _build_basis(X[:, 0], max_poly, n_fourier)
                out = Bx @ w
                return np.where(np.isfinite(out), out, 0.0)

            best_predict = predict_basis

    # --- Model 5: Exponential/Log Basis ---
    B_exp = _build_exp_basis(x)
    p_exp = B_exp.shape[1]
    for lam in lam_candidates:
        if time.time() - t_start > time_budget:
            break
        cv = _loocv_ridge(B_exp, y, lam)
        if cv < best_cv:
            best_cv = cv
            A = B_exp.T @ B_exp + lam * np.eye(p_exp)
            try:
                w = np.linalg.solve(A, B_exp.T @ y)
            except np.linalg.LinAlgError:
                continue

            def predict_exp(X, w=w):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                Bx = _build_exp_basis(X[:, 0])
                out = Bx @ w
                return np.where(np.isfinite(out), out, 0.0)

            best_predict = predict_exp

    # --- Model 6: Rational Basis ---
    B_rat = _build_rational_basis(x)
    p_rat = B_rat.shape[1]
    for lam in lam_candidates:
        if time.time() - t_start > time_budget:
            break
        cv = _loocv_ridge(B_rat, y, lam)
        if cv < best_cv:
            best_cv = cv
            A = B_rat.T @ B_rat + lam * np.eye(p_rat)
            try:
                w = np.linalg.solve(A, B_rat.T @ y)
            except np.linalg.LinAlgError:
                continue

            def predict_rat(X, w=w):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                Bx = _build_rational_basis(X[:, 0])
                out = Bx @ w
                return np.where(np.isfinite(out), out, 0.0)

            best_predict = predict_rat

    # --- Model 7: Tanh Basis ---
    B_tanh = _build_tanh_basis(x)
    p_tanh = B_tanh.shape[1]
    for lam in lam_candidates:
        if time.time() - t_start > time_budget:
            break
        cv = _loocv_ridge(B_tanh, y, lam)
        if cv < best_cv:
            best_cv = cv
            A = B_tanh.T @ B_tanh + lam * np.eye(p_tanh)
            try:
                w = np.linalg.solve(A, B_tanh.T @ y)
            except np.linalg.LinAlgError:
                continue

            def predict_tanh(X, w=w):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                Bx = _build_tanh_basis(X[:, 0])
                out = Bx @ w
                return np.where(np.isfinite(out), out, 0.0)

            best_predict = predict_tanh

    # --- Model 8: Hinge Basis ---
    # Use quantiles of x as thresholds
    thresholds = np.quantile(x, np.linspace(0.1, 0.9, 9))
    B_hinge = _build_hinge_basis(x, thresholds)
    p_hinge = B_hinge.shape[1]
    for lam in lam_candidates:
        if time.time() - t_start > time_budget:
            break
        cv = _loocv_ridge(B_hinge, y, lam)
        if cv < best_cv:
            best_cv = cv
            A = B_hinge.T @ B_hinge + lam * np.eye(p_hinge)
            try:
                w = np.linalg.solve(A, B_hinge.T @ y)
            except np.linalg.LinAlgError:
                continue

            def predict_hinge(X, w=w, thresholds=thresholds):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                Bx = _build_hinge_basis(X[:, 0], thresholds)
                out = Bx @ w
                return np.where(np.isfinite(out), out, 0.0)

            best_predict = predict_hinge

    # Fallback
    if best_predict is None:
        def predict_linear(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], np.mean(y))
        best_predict = predict_linear

    return best_predict
# EVOLVE-BLOCK-END
