"""Symbolic regression via adaptive multi-family basis with CV model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline


def _time_left(t0, deadline):
    return deadline - time.time() > 0


def _poly_features(x, degree):
    """Feature matrix [1, x, x^2, ..., x^degree]."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    F = np.empty((n, degree + 1))
    F[:, 0] = 1.0
    for i in range(1, degree + 1):
        F[:, i] = F[:, i - 1] * x
    return F


def _fourier_features(x, freqs):
    """Feature matrix [1, sin(w x), cos(w x) for each w in freqs]."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    cols = [np.ones(n)]
    for w in freqs:
        cols.append(np.sin(w * x))
        cols.append(np.cos(w * x))
    return np.column_stack(cols)


def _exp_features(x, rates):
    """Feature matrix [1, exp(r x) for r in rates]."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    x_safe = np.clip(x, -50.0, 50.0)
    cols = [np.ones(n)]
    for r in rates:
        cols.append(np.exp(r * x_safe))
    return np.column_stack(cols)


def _log_features(x):
    """Feature matrix [1, log|x|, (log|x|)^2]. Only valid if |x|>eps everywhere."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    ax = np.abs(x)
    lx = np.log(np.clip(ax, 1e-12, None))
    return np.column_stack([np.ones(n), lx, lx * lx])


def _recip_features(x):
    """Feature matrix [1, 1/x, 1/x^2]."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    safe = np.where(np.abs(x) > 1e-9, x, 1.0)
    inv = 1.0 / safe
    return np.column_stack([np.ones(n), inv, inv * inv])


def _sigmoid_features(x, scales):
    """Feature matrix [1, 1/(1+exp(-s x)) for s in scales]."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    x_safe = np.clip(x, -50.0, 50.0)
    cols = [np.ones(n)]
    for s in scales:
        cols.append(1.0 / (1.0 + np.exp(-s * x_safe)))
    return np.column_stack(cols)


def _sqrt_features(x):
    """Feature matrix [1, sqrt|x|, |x|^0.5 variants]."""
    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    ax = np.abs(x)
    return np.column_stack([np.ones(n), np.sqrt(ax), np.sign(x) * np.sqrt(ax)])


def _ridge_fit(F, y, alpha):
    """Solve (F^T F + alpha I) w = F^T y. Returns w."""
    n, p = F.shape
    A = F.T @ F
    A[np.diag_indices(p)] += alpha
    b = F.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(F, y, rcond=1e-12)[0]


def _cv_score(F, y, alpha, k=5):
    """K-fold CV normalized MSE for ridge fit of F @ w to y."""
    n = F.shape[0]
    if n < 2 * k:
        k = max(2, n // 5)
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    errs = []
    for i in range(k):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w = _ridge_fit(F[tr], y[tr], alpha)
        pred = F[val] @ w
        errs.append(np.mean((pred - y[val]) ** 2))
    mse = np.mean(errs)
    var_y = np.var(y)
    return mse / var_y if var_y > 1e-15 else 0.0


def fit(X_train, y_train):
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill
    n, d = X_train.shape
    x = X_train[:, 0].astype(np.float64)
    y = y_train.astype(np.float64)
    y_mean = float(np.mean(y))
    yc = y - y_mean

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = x_max - x_min
    all_positive = x_min > 1e-6
    has_zero = np.any(np.abs(x) < 1e-6)

    # ---- Build candidate feature families (only those safe for this range) ----
    candidates = []  # list of (name, F, alpha)

    # 1) Polynomial, several degrees
    for deg in [1, 2, 3, 4, 5, 6, 8]:
        F = _poly_features(x, deg)
        for alpha in [1e-6, 1e-4, 1e-3, 0.01, 0.1]:
            candidates.append((f"poly{deg}", F, alpha))

    # 2) Fourier with a frequency grid scaled to the span
    #    frequencies roughly cover 0.5..8 cycles over the span
    base_freq = 2.0 * np.pi / span if span > 1e-9 else 1.0
    freqs = [base_freq * m for m in np.arange(0.5, 8.5, 0.5)]
    F = _fourier_features(x, freqs)
    for alpha in [1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0]:
        candidates.append(("fourier", F, alpha))

    # 3) Exponential rates
    rates = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]
    F = _exp_features(x, rates)
    for alpha in [1e-6, 1e-4, 1e-3, 0.01, 0.1]:
        candidates.append(("exp", F, alpha))

    # 4) Log (only if strictly positive)
    if all_positive:
        F = _log_features(x)
        for alpha in [1e-6, 1e-4, 1e-3, 0.01]:
            candidates.append(("log", F, alpha))

    # 5) Reciprocal (only if no zero)
    if not has_zero:
        F = _recip_features(x)
        for alpha in [1e-6, 1e-4, 1e-3, 0.01]:
            candidates.append(("recip", F, alpha))

    # 6) Sigmoid
    scales = [0.5, 1.0, 2.0, 4.0]
    F = _sigmoid_features(x, scales)
    for alpha in [1e-6, 1e-4, 1e-3, 0.01]:
        candidates.append(("sigmoid", F, alpha))

    # 7) sqrt
    F = _sqrt_features(x)
    for alpha in [1e-6, 1e-4, 1e-3]:
        candidates.append(("sqrt", F, alpha))

    # ---- Evaluate candidates with CV, track best ----
    best_name = "poly1"
    best_F = _poly_features(x, 1)
    best_alpha = 1e-3
    best_cv = np.inf

    for name, F, alpha in candidates:
        if not _time_left(t0, deadline):
            break
        try:
            cv = _cv_score(F, yc, alpha)
        except Exception:
            continue
        if cv < best_cv:
            best_cv = cv
            best_name = name
            best_F = F
            best_alpha = alpha

    # ---- Fit the best candidate on all data ----
    try:
        w = _ridge_fit(best_F, yc, best_alpha)
    except Exception:
        w = np.linalg.lstsq(best_F, yc, rcond=1e-12)[0]

    # ---- Spline as a flexible fallback; compare CV ----
    spline_ok = False
    spline_obj = None
    try:
        if n >= 4:
            xs_sorted = np.sort(x)
            # unique-ish knots
            uniq = np.unique(np.round(xs_sorted, 8))
            if len(uniq) >= 4:
                # fit a cubic spline through sorted data (interpolating)
                order_y = np.argsort(x)
                spline_obj = CubicSpline(xs_sorted[order_y], yc[order_y][order_y])
                # CV score for spline: leave-folds-out interpolation
                k = 5
                idx = np.arange(n)
                folds = np.array_split(idx, k)
                errs = []
                for i in range(k):
                    val = folds[i]
                    tr = np.concatenate([folds[j] for j in range(k) if j != i])
                    xs_tr = np.sort(x[tr])
                    ys_tr = yc[tr][np.argsort(x[tr])]
                    uniq_tr = np.unique(np.round(xs_tr, 8))
                    if len(uniq_tr) >= 4:
                        sp_tr = CubicSpline(xs_tr, ys_tr)
                        pred = sp_tr(x[val])
                        errs.append(np.mean((pred - yc[val]) ** 2))
                if errs:
                    mse = np.mean(errs)
                    var_y = np.var(y)
                    cv_sp = mse / var_y if var_y > 1e-15 else 0.0
                    if cv_sp < best_cv:
                        best_cv = cv_sp
                        best_name = "spline"
                        spline_ok = True
    except Exception:
        spline_ok = False

    # ---- Finalize ----
    if spline_ok and spline_obj is not None:
        def predict(X):
            xp = np.asarray(X, dtype=np.float64).reshape(-1, X.shape[-1])[:, 0]
            out = spline_obj(xp)
            return out + y_mean
    else:
        F_best = best_F
        w_best = w

        def predict(X):
            xp = np.asarray(X, dtype=np.float64).reshape(-1, X.shape[-1])[:, 0]
            # Rebuild the same feature matrix as best_F
            # We need to know which family best_F came from.
            # Reconstruct by name.
            Fp = _rebuild(best_name, xp)
            return Fp @ w_best + y_mean

    return predict


def _rebuild(name, xp):
    """Rebuild the feature matrix for the given family name on xp."""
    x_min = float(np.min(xp))
    x_max = float(np.max(xp))
    span = x_max - x_min
    all_positive = x_min > 1e-6
    has_zero = np.any(np.abs(xp) < 1e-6)

    if name.startswith("poly"):
        deg = int(name[4:])
        return _poly_features(xp, deg)
    if name == "fourier":
        base_freq = 2.0 * np.pi / span if span > 1e-9 else 1.0
        freqs = [base_freq * m for m in np.arange(0.5, 8.5, 0.5)]
        return _fourier_features(xp, freqs)
    if name == "exp":
        rates = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]
        return _exp_features(xp, rates)
    if name == "log":
        return _log_features(xp)
    if name == "recip":
        return _recip_features(xp)
    if name == "sigmoid":
        scales = [0.5, 1.0, 2.0, 4.0]
        return _sigmoid_features(xp, scales)
    if name == "sqrt":
        return _sqrt_features(xp)
    # fallback
    return _poly_features(xp, 1)
# EVOLVE-BLOCK-END
