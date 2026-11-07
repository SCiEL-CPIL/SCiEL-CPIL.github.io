"""Symbolic regression: wavelet denoising + adaptive basis selection with fast CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 280.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _fit_rbf(X, y, sigma):
    """Fit Gaussian RBF model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2.0 * sigma**2))

    reg = 1e-6
    K_reg = K + reg * np.eye(n)

    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        diff_new = xn[:, None] - x[None, :]
        K_new = np.exp(-diff_new**2 / (2.0 * sigma**2))
        return K_new @ w

    return predict


def _fit_poly(X, y, deg):
    """Fit polynomial model with scaled input."""
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


def _fit_fourier(X, y, n_terms):
    """Fit Fourier series model."""
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


def _fit_spline(X, y, s_factor):
    """Fit B-spline model."""
    from scipy.interpolate import UnivariateSpline

    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

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


def _fit_ridge_rich(X, y, alpha):
    """Fit ridge regression on a rich feature basis."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    # Build rich feature matrix
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    # Polynomial terms
    x_scaled = (x - x.mean()) / x_range
    feats = [np.ones(n)]
    for d in range(1, 7):
        feats.append(x_scaled**d)

    # Fourier terms
    for k in range(1, 5):
        freq = 2.0 * np.pi * k / x_range
        feats.append(np.sin(freq * x))
        feats.append(np.cos(freq * x))

    # Exponential decay terms
    for lam in [0.5, 1.0, 2.0]:
        feats.append(np.exp(-lam * np.abs(x - x.mean()) / x_range))

    # Rational-like terms
    mask = np.abs(x) > 1e-10
    if np.sum(mask) > n // 2:
        feats.append(np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0))

    F = np.column_stack(feats)

    # Ridge regression
    try:
        A = F.T @ F + alpha * np.eye(F.shape[1])
        b = F.T @ y
        w = np.linalg.solve(A, b)
    except Exception:
        try:
            w = np.linalg.lstsq(F, y, rcond=None)[0]
        except Exception:
            w = np.zeros(F.shape[1])

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x.mean()) / x_range
        feats_new = [np.ones(len(xn))]
        for d in range(1, 7):
            feats_new.append(xn_scaled**d)
        for k in range(1, 5):
            freq = 2.0 * np.pi * k / x_range
            feats_new.append(np.sin(freq * xn))
            feats_new.append(np.cos(freq * xn))
        for lam in [0.5, 1.0, 2.0]:
            feats_new.append(np.exp(-lam * np.abs(xn - x.mean()) / x_range))
        mask_new = np.abs(xn) > 1e-10
        if np.sum(mask_new) > len(xn) // 2:
            feats_new.append(np.where(np.abs(xn) > 1e-10, 1.0 / xn, 0.0))
        Fn = np.column_stack(feats_new)
        return Fn @ w

    return predict


def _wavelet_denoise(X, y, level=4, threshold_factor=1.0):
    """
    Apply wavelet-based denoising to training data.
    Returns denoised y values at the same x locations.
    Uses a simple DWT implementation with Daubechies-4-like filters.
    """
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # For wavelet transform, we need equally-spaced data.
    # Interpolate onto a uniform grid first.
    n_grid = 256
    x_grid = np.linspace(x_sorted[0], x_sorted[-1], n_grid)
    y_grid = np.interp(x_grid, x_sorted, y_sorted)

    # Simple DWT with Haar-like filter (approximating Daubechies-4 behavior)
    # We'll use a simple iterative downsampling approach

    def dwt_step(signal):
        """One level of DWT with a 4-tap filter."""
        n = len(signal)
        if n < 4:
            return signal, signal

        # Daubechies-4 filter coefficients
        h0 = 0.4829629131445341
        h1 = 0.8365163037378079
        h2 = 0.2241438680420134
        h3 = -0.1294095225512604

        # Pad signal
        padded = np.concatenate([[signal[0]], signal, [signal[-1]]])

        # Approximation coefficients
        approx = []
        detail = []
        for i in range(0, n - 2, 2):
            a = h0 * padded[i] + h1 * padded[i + 1] + h2 * padded[i + 2] + h3 * padded[i + 3]
            d = h3 * padded[i] - h2 * padded[i + 1] + h1 * padded[i + 2] - h0 * padded[i + 3]
            approx.append(a)
            detail.append(d)

        return np.array(approx), np.array(detail)

    def idwt_step(approx, detail):
        """One level of inverse DWT."""
        n_a = len(approx)
        n_d = len(detail)
        n_out = 2 * max(n_a, n_d)

        # Daubechies-4 filter coefficients
        h0 = 0.4829629131445341
        h1 = 0.8365163037378079
        h2 = 0.2241438680420134
        h3 = -0.1294095225512604

        out = np.zeros(n_out)
        for i in range(n_a):
            idx = 2 * i
            if idx < n_out:
                out[idx] += h0 * approx[i]
            if idx + 1 < n_out:
                out[idx + 1] += h1 * approx[i]
            if idx + 2 < n_out:
                out[idx + 2] += h2 * approx[i]
            if idx + 3 < n_out:
                out[idx + 3] += h3 * approx[i]

        for i in range(n_d):
            idx = 2 * i
            if idx < n_out:
                out[idx] += h3 * detail[i]
            if idx + 1 < n_out:
                out[idx + 1] -= h2 * detail[i]
            if idx + 2 < n_out:
                out[idx + 2] += h1 * detail[i]
            if idx + 3 < n_out:
                out[idx + 3] -= h0 * detail[i]

        # Normalize
        out /= np.sqrt(2)
        return out[:n_out]

    # Forward transform
    coeffs = [y_grid]
    for _ in range(level):
        approx, detail = dwt_step(coeffs[-1])
        if len(approx) < 4:
            break
        coeffs.append(approx)
        coeffs.append(detail)

    # Soft thresholding on detail coefficients
    # Estimate noise level from finest detail
    finest_detail = coeffs[2 * level - 1] if len(coeffs) > 2 * level - 1 else coeffs[-1]
    if len(finest_detail) > 0:
        sigma_est = np.median(np.abs(finest_detail)) / 0.6745
    else:
        sigma_est = np.std(y_grid) / 10

    threshold = threshold_factor * sigma_est * np.sqrt(2 * np.log(n_grid))

    # Apply soft thresholding to all detail coefficients
    for i in range(2, len(coeffs), 2):
        if i < len(coeffs):
            coeffs[i] = np.sign(coeffs[i]) * np.maximum(np.abs(coeffs[i]) - threshold, 0)

    # Inverse transform
    reconstructed = coeffs[0].copy()
    for i in range(len(coeffs) - 1, 1, -2):
        if i >= 2 and i - 1 >= 0:
            reconstructed = idwt_step(coeffs[i - 1], coeffs[i])

    # Interpolate back to original x locations
    y_denoised_grid = reconstructed[:n_grid]
    y_denoised = np.interp(x_sorted, x_grid, y_denoised_grid)

    # Unsort
    y_final = np.empty(n)
    y_final[sort_idx] = y_denoised
    return y_final


def _kfold_cv_error(fit_func, X, y, k=5, seed=42):
    """Calculate K-Fold CV MSE."""
    n = len(X)
    if n < k + 2:
        k = max(2, n - 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    errs = []
    for f in folds:
        if _remaining() < 5:
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

    # Try wavelet denoised version
    try:
        y_denoised = _wavelet_denoise(X, y, level=3, threshold_factor=0.8)
    except Exception:
        y_denoised = y.copy()

    # Candidate models: use both original and denoised y
    candidates = []

    # RBF with different sigmas (on original y)
    sigma_candidates = [0.05 * x_range, 0.1 * x_range, 0.2 * x_range, 0.3 * x_range,
                        0.5 * x_range, 0.8 * x_range, 1.2 * x_range]
    for sigma in sigma_candidates:
        candidates.append((f"RBF_orig(s={sigma:.3f})",
                           lambda X_tr, y_tr, s=sigma: _fit_rbf(X_tr, y_tr, s), y))
        candidates.append((f"RBF_wv(s={sigma:.3f})",
                           lambda X_tr, y_tr, s=sigma: _fit_rbf(X_tr, y_tr, s), y_denoised))

    # Polynomials
    for deg in [2, 3, 4, 5, 6, 8]:
        candidates.append((f"Poly_orig(d={deg})",
                           lambda X_tr, y_tr, d=deg: _fit_poly(X_tr, y_tr, d), y))
        candidates.append((f"Poly_wv(d={deg})",
                           lambda X_tr, y_tr, d=deg: _fit_poly(X_tr, y_tr, d), y_denoised))

    # Fourier series
    for n_terms in [2, 3, 4, 5, 8]:
        candidates.append((f"Fourier_orig(n={n_terms})",
                           lambda X_tr, y_tr, nt=n_terms: _fit_fourier(X_tr, y_tr, nt), y))
        candidates.append((f"Fourier_wv(n={n_terms})",
                           lambda X_tr, y_tr, nt=n_terms: _fit_fourier(X_tr, y_tr, nt), y_denoised))

    # Splines
    for sf in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((f"Spline_orig(s={sf})",
                           lambda X_tr, y_tr, f=sf: _fit_spline(X_tr, y_tr, f), y))
        candidates.append((f"Spline_wv(s={sf})",
                           lambda X_tr, y_tr, f=sf: _fit_spline(X_tr, y_tr, f), y_denoised))

    # Ridge on rich basis
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        candidates.append((f"Ridge_orig(a={alpha})",
                           lambda X_tr, y_tr, a=alpha: _fit_ridge_rich(X_tr, y_tr, a), y))
        candidates.append((f"Ridge_wv(a={alpha})",
                           lambda X_tr, y_tr, a=alpha: _fit_ridge_rich(X_tr, y_tr, a), y_denoised))

    # Evaluate candidates using K-Fold CV
    best_model = None
    best_cv_err = np.inf
    best_name = "none"

    # Sort candidates to try most promising first
    # RBF and splines tend to work well for smooth functions
    priority_order = {
        'RBF': 0, 'Spline': 1, 'Ridge': 2, 'Poly': 3, 'Fourier': 4
    }

    def _priority(name):
        for key, val in priority_order.items():
            if name.startswith(key):
                return val
        return 5

    candidates.sort(key=lambda c: _priority(c[0]))

    for name, fit_func, y_data in candidates:
        if _remaining() < 10:
            break
        try:
            cv_err = _kfold_cv_error(fit_func, X, y_data, k=5)
            if cv_err < best_cv_err:
                best_cv_err = cv_err
                best_model = fit_func(X, y_data)
                best_name = name
        except Exception:
            pass

    # If no model was selected, use a simple fallback
    if best_model is None:
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            best_model = predict
            best_name = "Linear"
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            best_model = predict
            best_name = "Constant"

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
