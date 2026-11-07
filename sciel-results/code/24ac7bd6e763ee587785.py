# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X, d):
    """Build a rich set of candidate basis functions evaluated on X. Returns (N, p) array."""
    N = X.shape[0]
    cols = []

    # Constant
    cols.append(np.ones(N))

    # Linear
    for j in range(d):
        cols.append(X[:, j])

    # Squares through 6th power
    for j in range(d):
        for pwr in range(2, 7):
            cols.append(X[:, j] ** pwr)

    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])

    # Triple products
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])

    # Trig: sin/cos of x, 2x, 3x
    for j in range(d):
        xj = X[:, j]
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        cols.append(np.sin(2.0 * xj))
        cols.append(np.cos(2.0 * xj))
        cols.append(np.sin(3.0 * xj))
        cols.append(np.cos(3.0 * xj))

    # Exp, log, sqrt, sigmoid, tanh
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        cols.append(np.exp(-xj))
        cols.append(np.exp(2.0 * xj))
        cols.append(np.exp(-2.0 * xj))
        cols.append(np.log1p(xj))
        cols.append(np.sqrt(xj))
        cols.append(np.tanh(xj))
        cols.append(np.tanh(2.0 * xj))
        cols.append(1.0 / (1.0 + np.exp(-xj)))

    # Cross terms with trig and exp
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                xi, xj = X[:, i], X[:, j]
                cols.append(np.sin(xi) * xj)
                cols.append(np.cos(xi) * xj)
                cols.append(np.sin(xi) * np.sin(xj))
                cols.append(np.cos(xi) * np.cos(xj))
                cols.append(xi * np.exp(xj))
                cols.append(np.exp(xi) * xj)
                cols.append(xi * np.sqrt(xj))
                cols.append(np.sqrt(xi) * xj)
                cols.append(np.sin(xi) * np.exp(xj))
                cols.append(np.cos(xi) * np.exp(xj))

    # Global features
    cols.append(np.sum(X, axis=1))
    cols.append(np.prod(X, axis=1))
    cols.append(np.sum(X ** 2, axis=1))
    cols.append(np.sum(X ** 3, axis=1))
    cols.append(np.linalg.norm(X, axis=1))
    cols.append(np.linalg.norm(X, axis=1) ** 2)
    cols.append(np.min(X, axis=1))
    cols.append(np.max(X, axis=1))

    # Differences
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] - X[:, j])
                cols.append(X[:, i] * (X[:, i] - X[:, j]))

    return np.column_stack(cols)


def _orthogonal_matching_pursuit(F, y, max_terms=15, tol=1e-6):
    """
    OMP with ridge refit at the end.
    Returns (active_indices, w_active).
    """
    n, p = F.shape
    residual = y.copy()
    active = []
    proj = np.zeros(n)

    for _ in range(max_terms):
        if _time_left() < 2.0:
            break
        # Find feature with max correlation to residual
        correlations = F.T @ residual
        # Mask already active
        for j in active:
            correlations[j] = -np.inf
        if np.all(np.isinf(correlations)):
            break
        j = int(np.argmax(np.abs(correlations)))
        if abs(correlations[j]) < tol:
            break
        active.append(j)

        # Refit on active set using least squares
        F_active = F[:, active]
        try:
            w_active = np.linalg.lstsq(F_active, y, rcond=None)[0]
            proj = F_active @ w_active
            residual = y - proj
        except Exception:
            break

    if len(active) == 0:
        active = [0]
        F_active = F[:, active]
        try:
            w_active = np.linalg.lstsq(F_active, y, rcond=None)[0]
        except Exception:
            w_active = np.array([np.mean(y)])

    return active, w_active


def _fit_ridge(F, y, alpha=1e-8):
    n, p = F.shape
    A = F.T @ F + alpha * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _rbf_predict(X_train, y_train, X, gamma, alpha=1e-6):
    Xa = X_train
    Xb = X
    sq_a = np.sum(Xa ** 2, axis=1)[:, None]
    sq_b = np.sum(Xb ** 2, axis=1)[None, :]
    K_train = np.exp(-gamma * (sq_a + sq_a.T - 2.0 * Xa @ Xa.T))
    K_test = np.exp(-gamma * (sq_a + sq_b - 2.0 * Xa @ X.T))

    n = X_train.shape[0]
    A = K_train + alpha * np.eye(n)
    try:
        theta = np.linalg.solve(A, y_train)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, y_train, rcond=None)[0]
    return K_test @ theta


def _gaussian_process_predict(X_train, y_train, X, length_scale, noise_var=1e-6):
    """Simple GP prediction with RBF kernel."""
    # Normalize inputs
    Xa = X_train
    Xb = X

    # Compute kernel matrices
    def rbf_kernel(X1, X2, ls):
        sq1 = np.sum(X1 ** 2, axis=1)[:, None]
        sq2 = np.sum(X2 ** 2, axis=1)[None, :]
        sq_dist = sq1 + sq2 - 2.0 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0)
        return np.exp(-sq_dist / (2.0 * ls ** 2))

    K_train = rbf_kernel(Xa, Xa, length_scale)
    K_test = rbf_kernel(Xa, Xb, length_scale)

    n = X_train.shape[0]
    A = K_train + noise_var * np.eye(n)
    try:
        L = np.linalg.cholesky(A)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_test @ alpha
    except np.linalg.LinAlgError:
        try:
            theta = np.linalg.solve(A, y_train)
            mu = K_test @ theta
        except Exception:
            mu = np.full(Xb.shape[0], np.mean(y_train))
    return mu


def _try_polynomial_fit(X, y, d, max_degree=4):
    """Try polynomial regression with increasing degrees."""
    N = X.shape[0]
    best_w = None
    best_mse = np.inf
    best_degree = 1

    for degree in range(1, max_degree + 1):
        if _time_left() < 3.0:
            break
        # Build polynomial features
        from itertools import combinations_with_replacement
        cols = [np.ones(N)]
        for idx in combinations_with_replacement(range(d), degree):
            term = np.ones(N)
            for j in idx:
                term = term * X[:, j]
            cols.append(term)
        F = np.column_stack(cols)
        if F.shape[1] > N:
            continue
        try:
            w = np.linalg.lstsq(F, y, rcond=None)[0]
            pred = F @ w
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w
                best_degree = degree
                best_F_shape = F.shape[1]
        except Exception:
            continue

    return best_mse, best_degree, best_w


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    if N == 0:
        return lambda X: np.zeros(np.asarray(X).shape[0])

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-12:
        return lambda X: np.full(np.asarray(X).shape[0], y_mean)

    # Build basis
    F = _build_basis(X_train, d)
    p = F.shape[1]

    # Ensure finite
    finite_cols = np.all(np.isfinite(F), axis=0)
    F = F[:, finite_cols]
    p = F.shape[1]
    finite_mask = finite_cols

    if p == 0:
        return lambda X: np.full(np.asarray(X).shape[0], y_mean)

    # Standardize features (keep constant)
    F_mean = np.mean(F, axis=0)
    F_std = np.std(F, axis=0)
    F_std[F_std < 1e-12] = 1.0
    F_std[0] = 1.0
    F_mean[0] = 0.0
    F_norm = (F - F_mean) / F_std

    # y stats
    y_norm = (y_train - y_mean) / y_std

    # Strategy 1: OMP
    active_omp, w_omp = _orthogonal_matching_pursuit(F_norm, y_norm, max_terms=20, tol=1e-5)
    F_active_omp = F_norm[:, active_omp]
    w_omp_refit = _fit_ridge(F_active_omp, y_norm, alpha=1e-8)
    pred_omp = F_active_omp @ w_omp_refit
    mse_omp = np.mean((pred_omp - y_norm) ** 2)

    # Strategy 2: RBF
    best_mse_rbf = np.inf
    best_gamma = None
    X_scale = np.std(X_train, axis=0)
    X_scale[X_scale < 1e-12] = 1.0
    d_eff = np.sqrt(np.sum(X_scale ** 2))
    gamma_candidates = [1.0 / (2.0 * d * d_eff ** 2),
                        1.0 / (d_eff ** 2),
                        1.0 / (d_eff ** 2) * 5.0,
                        1.0 / (d_eff ** 2) * 0.2,
                        1.0 / (d_eff ** 2) * 10.0,
                        1.0 / (d_eff ** 2) * 0.1]
    for gamma in gamma_candidates:
        if _time_left() < 5.0:
            break
        try:
            pred_rbf = _rbf_predict(X_train, y_train, X_train, gamma, alpha=1e-6)
            mse_rbf = np.mean((pred_rbf - y_train) ** 2) / (y_std ** 2)
            if mse_rbf < best_mse_rbf:
                best_mse_rbf = mse_rbf
                best_gamma = gamma
        except Exception:
            continue

    # Strategy 3: GP
    best_mse_gp = np.inf
    best_ls = None
    ls_candidates = [0.5, 1.0, 2.0, 0.25, 4.0]
    for ls in ls_candidates:
        if _time_left() < 5.0:
            break
        try:
            pred_gp = _gaussian_process_predict(X_train, y_train, X_train, ls, noise_var=1e-6)
            mse_gp = np.mean((pred_gp - y_train) ** 2) / (y_std ** 2)
            if mse_gp < best_mse_gp:
                best_mse_gp = mse_gp
                best_ls = ls
        except Exception:
            continue

    # Strategy 4: Polynomial
    mse_poly, deg_poly, w_poly = _try_polynomial_fit(X_train, y_train, d, max_degree=4)
    mse_poly_norm = mse_poly / (y_std ** 2)

    # Choose best strategy
    candidates = [
        ("omp", mse_omp, active_omp, w_omp_refit),
        ("rbf", best_mse_rbf, best_gamma, None),
        ("gp", best_mse_gp, best_ls, None),
        ("poly", mse_poly_norm, deg_poly, w_poly),
    ]

    best_type = "omp"
    best_mse = mse_omp
    best_active = active_omp
    best_w = w_omp_refit
    best_gamma = None
    best_ls = None
    best_deg = None
    best_w_poly = None

    for name, mse, param1, param2 in candidates:
        if np.isfinite(mse) and mse < best_mse:
            best_mse = mse
            best_type = name
            if name == "omp":
                best_active = param1
                best_w = param2
            elif name == "rbf":
                best_gamma = param1
            elif name == "gp":
                best_ls = param1
            elif name == "poly":
                best_deg = param1
                best_w_poly = param2

    # Capture
    F_mean_c = F_mean.copy()
    F_std_c = F_std.copy()
    sel_indices = best_active if best_type == "omp" else []
    w_c = best_w if best_type == "omp" else None
    y_mean_c = y_mean
    y_std_c = y_std
    best_type_c = best_type
    best_gamma_c = best_gamma
    best_ls_c = best_ls
    best_deg_c = best_deg
    best_w_poly_c = best_w_poly
    X_train_c = X_train.copy()
    y_train_c = y_train.copy()
    finite_mask_c = finite_mask.copy()
    d_c = d

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if best_type_c == "rbf" and best_gamma_c is not None:
            return _rbf_predict(X_train_c, y_train_c, X, best_gamma_c, alpha=1e-6)
        elif best_type_c == "gp" and best_ls_c is not None:
            return _gaussian_process_predict(X_train_c, y_train_c, X, best_ls_c, noise_var=1e-6)
        elif best_type_c == "poly" and best_w_poly_c is not None:
            from itertools import combinations_with_replacement
            Np = X.shape[0]
            cols = [np.ones(Np)]
            for idx in combinations_with_replacement(range(d_c), best_deg_c):
                term = np.ones(Np)
                for j in idx:
                    term = term * X[:, j]
                cols.append(term)
            Fp = np.column_stack(cols)
            return Fp @ best_w_poly_c
        else:
            F_pred = _build_basis(X, d_c)
            F_pred = F_pred[:, finite_mask_c]
            F_pred_norm = (F_pred - F_mean_c) / F_std_c
            F_pred_sel = F_pred_norm[:, sel_indices]
            return F_pred_sel @ w_c * y_std_c + y_mean_c

    return predict
# EVOLVE-BLOCK-END
