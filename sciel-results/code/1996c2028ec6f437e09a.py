"""Rewritten program: GPR-smoothed degree-5 polynomial closure for two-scale L96."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        # Not enough points for degree-5 fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n - 1))
            full = np.zeros(6)
            full[:len(coeff)] = coeff
            return full
        except Exception:
            return np.zeros(6)

    # --- Stage 1: GPR denoising to smooth the training data ---
    # Use a Gaussian process regression with a squared-exponential kernel
    # to denoise the (x, y) pairs, then fit a degree-5 polynomial to the
    # smoothed values. This reduces overfitting to fast-scale noise.

    def gpr_smooth(x_train, y_train, x_eval, length_scale=0.5, noise_var=0.1):
        """Simple GPR with squared-exponential kernel."""
        # Normalize x for numerical stability
        x_mean = np.mean(x_train)
        x_std = np.std(x_train)
        if x_std < 1e-12:
            x_std = 1.0
        xn_train = (x_train - x_mean) / x_std
        xn_eval = (x_eval - x_mean) / x_std

        # Kernel matrix
        def sqexp(a, b, ls):
            return np.exp(-0.5 * ((a[:, None] - b[None, :]) ** 2) / (ls ** 2))

        K = sqexp(xn_train, xn_train, length_scale)
        K += noise_var * np.eye(len(xn_train))

        # Solve for weights
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            # Fallback to lstsq if Cholesky fails
            alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]

        # Predict at eval points
        Ks = sqexp(xn_eval, xn_train, length_scale)
        mu = Ks @ alpha
        return mu

    # Choose evaluation points on a fine grid
    x_min, x_max = np.min(x), np.max(x)
    n_grid = min(500, max(200, n))
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Try a few length scales and noise levels, pick the best via cross-validation
    best_smoothed = None
    best_cv_err = np.inf
    best_params = (0.5, 0.1)

    length_scales = [0.1, 0.2, 0.5, 1.0, 2.0]
    noise_vars = [0.01, 0.05, 0.1, 0.5, 1.0]

    # Use a subset for speed if n is large
    if n > 2000:
        idx = np.random.RandomState(42).choice(n, 2000, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Simple CV: split into 5 folds
    n_folds = 5
    fold_size = len(x_sub) // n_folds
    if fold_size < 6:
        n_folds = max(2, len(x_sub) // 6)
        fold_size = len(x_sub) // n_folds

    for ls in length_scales:
        for nv in noise_vars:
            if time.time() > deadline - 20:
                break
            cv_errors = []
            for fold in range(n_folds):
                val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_idx = np.concatenate([np.arange(0, fold * fold_size),
                                            np.arange((fold + 1) * fold_size, len(x_sub))])
                if len(train_idx) < 6:
                    continue
                x_tr = x_sub[train_idx]
                y_tr = y_sub[train_idx]
                x_val = x_sub[val_idx]

                # Smooth the training data
                x_smooth_grid = np.linspace(np.min(x_tr), np.max(x_tr), min(200, len(x_tr)))
                y_smoothed = gpr_smooth(x_tr, y_tr, x_smooth_grid, ls, nv)

                # Interpolate to validation points
                y_val_pred = np.interp(x_val, x_smooth_grid, y_smoothed)
                cv_errors.append(np.mean((y_val - y_val_pred) ** 2))

            if cv_errors:
                avg_err = np.mean(cv_errors)
                if avg_err < best_cv_err:
                    best_cv_err = avg_err
                    best_params = (ls, nv)

        if time.time() > deadline - 20:
            break

    ls_best, nv_best = best_params

    # Now smooth the full training data with best params
    if time.time() < deadline - 15:
        x_smooth_grid = np.linspace(x_min, x_max, n_grid)
        y_smoothed = gpr_smooth(x, y, x_smooth_grid, ls_best, nv_best)
        # Fit degree-5 polynomial to smoothed data
        try:
            coeff = np.polynomial.polynomial.polyfit(x_smooth_grid, y_smoothed, 5)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass

    # Fallback: direct degree-5 polyfit on raw data
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    # Last resort: lower degree
    for deg in range(4, 0, -1):
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full = np.zeros(6)
            full[:len(coeff)] = coeff
            if np.all(np.isfinite(full)):
                return full
        except Exception:
            continue

    return np.zeros(6)
# EVOLVE-BLOCK-END
