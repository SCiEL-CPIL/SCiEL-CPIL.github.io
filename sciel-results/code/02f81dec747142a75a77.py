"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson(x, y, x_eval, bandwidth):
    """Compute Nadaraya-Watson kernel regression estimate at x_eval points."""
    diff = x_eval[:, None] - x[None, :]
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
    wsum = np.sum(weights, axis=1)
    wsum[wsum < 1e-30] = 1e-30
    y_est = np.sum(weights * y[None, :], axis=1) / wsum
    return y_est


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 20:
        return np.zeros(6)

    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 20:
        return np.zeros(6)

    # Center and scale x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Split into train/val for candidate selection
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_frac = 0.2
    val_size = max(100, int(n * val_frac))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_train = x_scaled[train_idx]
    y_train = y[train_idx]
    x_val = x_scaled[val_idx]
    y_val = y[val_idx]

    def _score_candidate(coeff_scaled):
        """Compute normalized MSE on validation set."""
        pred_val = np.polynomial.polynomial.polyval(x_val, coeff_scaled)
        mse = np.mean((pred_val - y_val) ** 2)
        var_y = np.var(y_val)
        if var_y < 1e-10:
            return 1e10
        return mse / var_y

    def _convert_to_original(coeff_scaled, x_mean, x_std):
        """Convert coefficients from scaled to original x scale."""
        a = 1.0 / x_std
        b = -x_mean / x_std
        coeff = np.zeros(6)
        for i in range(6):
            for j in range(i + 1):
                binom = 1.0
                for k in range(j):
                    binom *= (i - k) / (k + 1)
                coeff[j] += coeff_scaled[i] * binom * (a ** j) * (b ** (i - j))
        return np.real(coeff)

    best_coeff = None
    best_score = np.inf

    # Strategy 1: Nadaraya-Watson denoising with various bandwidths
    bandwidths = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
    for bw in bandwidths:
        if time.time() > deadline - 20:
            break
        try:
            y_den = _nadaraya_watson(x_train, y_train, x_train, bw)
            coeff_scaled = np.polynomial.polynomial.polyfit(x_train, y_den, 5)
            if np.all(np.isfinite(coeff_scaled)):
                score = _score_candidate(coeff_scaled)
                if score < best_score:
                    best_score = score
                    best_coeff = coeff_scaled.copy()
        except:
            pass

    # Strategy 2: Ridge regression with different regularization strengths
    X_vander = np.vander(x_train, 6, increasing=True).T  # shape (n, 6)
    lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for lam in lambdas:
        if time.time() > deadline - 15:
            break
        try:
            A = X_vander.T @ X_vander + lam * np.eye(6)
            B = X_vander.T @ y_train
            coeff_scaled = np.linalg.solve(A, B)
            if np.all(np.isfinite(coeff_scaled)):
                score = _score_candidate(coeff_scaled)
                if score < best_score:
                    best_score = score
                    best_coeff = coeff_scaled.copy()
        except:
            pass

    # Strategy 3: Polynomial fit on raw (non-denoised) data
    try:
        coeff_scaled = np.polynomial.polynomial.polyfit(x_train, y_train, 5)
        if np.all(np.isfinite(coeff_scaled)):
            score = _score_candidate(coeff_scaled)
            if score < best_score:
                best_score = score
                best_coeff = coeff_scaled.copy()
    except:
        pass

    # Strategy 4: Weighted polynomial fit - weight points by density
    for bw in [0.2, 0.3, 0.5]:
        if time.time() > deadline - 10:
            break
        try:
            diff = x_train[:, None] - x_train[None, :]
            dens = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1)
            weights = dens / (dens.sum() + 1e-10)
            coeff_scaled = np.polynomial.polynomial.polyfit(x_train, y_train, 5, w=weights)
            if np.all(np.isfinite(coeff_scaled)):
                score = _score_candidate(coeff_scaled)
                if score < best_score:
                    best_score = score
                    best_coeff = coeff_scaled.copy()
        except:
            pass

    # Final safety check
    if best_coeff is None:
        # Fallback: simple polyfit on all data
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        except:
            return np.zeros(6)
    else:
        # Try fitting on the full dataset and compare
        try:
            coeff_full = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
            if np.all(np.isfinite(coeff_full)):
                pred_full = np.polynomial.polynomial.polyval(x_val, coeff_full)
                mse_full = np.mean((pred_full - y_val) ** 2)
                var_y = np.var(y_val)
                score_full = mse_full / (var_y + 1e-10)
                if score_full < best_score:
                    best_coeff = coeff_full.copy()
                    best_score = score_full
        except:
            pass

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        return np.zeros(6)

    # Convert to original scale
    coeff_original = _convert_to_original(best_coeff, x_mean, x_std)

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_original)):
        return np.zeros(6)

    return coeff_original

# EVOLVE-BLOCK-END
