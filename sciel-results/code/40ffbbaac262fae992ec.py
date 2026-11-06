"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)

    # Filter out non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 10:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # Normalize x for numerical stability in polynomial fitting
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Two-stage approach:
    # Stage 1: Nadaraya-Watson kernel smoothing to denoise the target
    # Stage 2: Fit degree-5 polynomial to smoothed values

    # Try multiple bandwidths via cross-validation on the smoothed fit
    # We'll use a grid of bandwidths in normalized x-space
    bandwidths = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0])

    best_cv_score = np.inf
    best_bandwidth = 0.3
    best_smoothed_y = None

    # For cross-validation, split into folds
    n_folds = 5
    indices = np.arange(n)
    fold_size = n // n_folds

    for bw in bandwidths:
        if time.time() > deadline - 5.0:
            break

        cv_errors = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            if fold == n_folds - 1:
                val_idx = indices[fold * fold_size:]  # last fold gets remainder
            train_idx = np.setdiff1d(indices, val_idx)

            if len(train_idx) < 10 or len(val_idx) < 5:
                continue

            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]

            # Nadaraya-Watson smoothing on training data
            weights = np.exp(-0.5 * ((x_val[:, None] - x_train[None, :]) / bw) ** 2)
            # Avoid division by zero
            weight_sums = np.sum(weights, axis=1)
            weight_sums[weight_sums < 1e-15] = 1e-15
            y_smoothed_val = np.sum(weights * y_train[None, :], axis=1) / weight_sums

            # Fit degree-5 polynomial to smoothed values
            try:
                # Use the smoothed validation y to fit? No - we need to fit on train smoothed
                # Actually for CV: fit polynomial on smoothed train, predict on val
                # But we need smoothed y for train too
                weights_train = np.exp(-0.5 * ((x_train[:, None] - x_train[None, :]) / bw) ** 2)
                weight_sums_train = np.sum(weights_train, axis=1)
                weight_sums_train[weight_sums_train < 1e-15] = 1e-15
                y_smoothed_train = np.sum(weights_train * y_train[None, :], axis=1) / weight_sums_train

                # Fit poly on smoothed train
                coeff_trial = np.polynomial.polynomial.polyfit(x_train, y_smoothed_train, 5)
                # Predict on val
                y_pred = np.polynomial.polynomial.polyval(x_val, coeff_trial)
                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)
            except Exception:
                continue

        if len(cv_errors) > 0:
            avg_cv = np.mean(cv_errors)
            if avg_cv < best_cv_score:
                best_cv_score = avg_cv
                best_bandwidth = bw

    # Final: smooth all training data with best bandwidth and fit polynomial
    bw = best_bandwidth
    weights_all = np.exp(-0.5 * ((x_norm[:, None] - x_norm[None, :]) / bw) ** 2)
    weight_sums_all = np.sum(weights_all, axis=1)
    weight_sums_all[weight_sums_all < 1e-15] = 1e-15
    y_smoothed = np.sum(weights_all * y[None, :], axis=1) / weight_sums_all

    # Fit degree-5 polynomial to smoothed data
    try:
        coeff_norm = np.polynomial.polynomial.polyfit(x_norm, y_smoothed, 5)
    except Exception:
        # Fallback to direct fit
        coeff_norm = np.polynomial.polynomial.polyfit(x_norm, y, 5)

    # Convert from normalized x back to original x
    # If p(x_norm) = sum c_i * x_norm^i, and x_norm = (x - x_mean)/x_std
    # Then p(x) = sum c_i * ((x - x_mean)/x_std)^i
    # We need to expand this into powers of x
    # Use composition: let u = (x - x_mean)/x_std = x/x_std - x_mean/x_std
    # p(u) = sum c_i * u^i
    # We want q(x) = p((x - x_mean)/x_std)
    # q(x) = sum c_i * (x/x_std - x_mean/x_std)^i
    # Use polynomial composition via np.polynomial

    from numpy.polynomial import polynomial as P

    # x_norm = (x - x_mean) / x_std = (1/x_std)*x + (-x_mean/x_std)
    # So x_norm as polynomial in x: [ -x_mean/x_std, 1/x_std ]
    x_norm_as_poly = np.array([-x_mean / x_std, 1.0 / x_std])

    # Compose: q(x) = p(x_norm_as_poly(x))
    # Use P.polyfromroots? No, use P.polycompos
    try:
        coeff_orig = P.polycompos(coeff_norm, x_norm_as_poly)
        # Ensure we have exactly 6 coefficients
        if len(coeff_orig) < 6:
            coeff_orig = np.pad(coeff_orig, (0, 6 - len(coeff_orig)))
        else:
            coeff_orig = coeff_orig[:6]
    except Exception:
        # Fallback: just use the normalized coefficients as-is (approximate)
        coeff_orig = np.zeros(6)
        coeff_orig[:min(6, len(coeff_norm))] = coeff_norm[:min(6, len(coeff_norm))]

    # Final sanity check: all coefficients must be finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.zeros(6)
        if n >= 6:
            try:
                coeff_direct = np.polynomial.polynomial.polyfit(x, y, 5)
                coeff_orig[:min(6, len(coeff_direct))] = coeff_direct[:min(6, len(coeff_direct))]
            except Exception:
                pass

    return coeff_orig
# EVOLVE-BLOCK-END
