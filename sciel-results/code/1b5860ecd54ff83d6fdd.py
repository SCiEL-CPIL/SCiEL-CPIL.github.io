"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Nadaraya-Watson smoothing to denoise the target ---
    # Estimate bandwidth using MAD
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad < 1e-12:
        mad = np.std(x)
    if mad < 1e-12:
        mad = 1.0
    # Silverman-like bandwidth scaled for the data range
    bw = 1.06 * max(mad, np.std(x) * 0.9) * n ** (-0.2)
    bw = max(bw, (x.max() - x.min()) / 20.0)  # ensure at least 20 bins worth

    # Evaluate NW kernel at each training point to get smoothed y
    # Use Gaussian kernel: K(u) = exp(-0.5 * u^2)
    # To avoid O(n^2) for very large n, subsample if needed
    if n > 50000:
        # Subsample for bandwidth evaluation, but use full data for regression
        idx = np.random.default_rng(42).choice(n, size=50000, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y
        idx = None

    # Compute smoothed values using subsampled points
    # Process in chunks to manage memory
    chunk_size = 5000
    y_smoothed = np.empty(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x[start:end]
        # u = (x_sub - x_chunk[:, None]) / bw
        u = (x_sub[None, :] - x_chunk[:, None]) / bw
        weights = np.exp(-0.5 * u * u)
        w_sum = np.sum(weights, axis=1)
        w_sum = np.where(w_sum < 1e-300, 1e-300, w_sum)
        y_smoothed[start:end] = np.sum(weights * y_sub[None, :], axis=1) / w_sum

    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: linear fit on smoothed data
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y_smoothed, 1)
        except Exception:
            coeff[1] = np.mean(y_smoothed) / np.mean(x) if np.mean(x) != 0 else 0
        return coeff

    # --- Degree-5 polynomial fit on smoothed data ---
    # Use polynomial fit; for numerical stability, center and scale x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    try:
        # Fit degree-5 polynomial on scaled x
        coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y_smoothed, 5)
        
        # Transform coefficients back to original x
        # y = sum(c_i * ((x - x_mean)/x_std)^i)
        # We need to express in powers of x
        # Use polynomial conversion: p_scaled(t) where t = (x - x_mean)/x_std
        # p(x) = p_scaled((x - x_mean)/x_std)
        # This is a polynomial in x of degree 5
        # Use numpy's polyfromroots or direct expansion
        # Simpler: use polyfit on original x directly (may be less stable but for degree 5 it's usually ok)
        
        # Actually, let's just fit on original x to avoid transformation complexity
        # For degree 5, direct fit is usually acceptable if x range isn't extreme
        coeff = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
    except Exception:
        # Fallback to lower degree
        try:
            coeff = np.zeros(6)
            coeff[:3] = np.polynomial.polynomial.polyfit(x, y_smoothed, 2)
        except Exception:
            coeff = np.zeros(6)
            coeff[1] = np.cov(x, y_smoothed, bias=True)[0, 1] / np.var(x) if np.var(x) > 1e-12 else 0
            coeff[0] = np.mean(y_smoothed) - coeff[1] * np.mean(x)

    # Ensure finite coefficients
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)

    return coeff
# EVOLVE-BLOCK-END
