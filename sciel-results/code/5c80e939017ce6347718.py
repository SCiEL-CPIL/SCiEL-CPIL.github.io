# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Nadaraya-Watson kernel smoothing to denoise ---
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    # Silverman's rule of thumb for bandwidth
    mad = np.median(np.abs(x - np.median(x)))
    if mad < 1e-12:
        mad = x_std * 0.6745
    bandwidth = 0.9 * min(x_std, 1.06 * mad / 0.6745) * n ** (-1.0 / 5.0)
    if bandwidth < 1e-6:
        bandwidth = x_std * 0.1

    # Evaluate smoothed values at training points using chunked vectorized kernel
    y_smoothed = np.empty_like(y)
    # Choose chunk size to keep memory reasonable (~200MB max for diff matrix)
    chunk_size = min(n, max(500, int(2e8 / (8 * n))))  # ~200MB / (8 bytes * n cols)

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        if time.time() > deadline - 10.0:
            y_smoothed[i_start:i_end] = y[i_start:i_end]
            break

        x_chunk = x[i_start:i_end]
        diff = x_chunk[:, np.newaxis] - x[np.newaxis, :]
        u = diff / bandwidth
        u = np.clip(u, -40.0, 40.0)
        weights = np.exp(-0.5 * u * u)

        numerator = weights @ y
        denominator = weights.sum(axis=1)
        denom_safe = np.where(denominator > 1e-12, denominator, 1.0)
        y_smoothed[i_start:i_end] = numerator / denom_safe

    # --- Fit degree-5 polynomial to smoothed data ---
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=6, increasing=True)

    best_coeff_raw = np.zeros(6)
    best_mse = np.inf

    lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
        A = V.T @ V
        b = V.T @ y_smoothed
        if lam > 0:
            A_reg = A + lam * np.eye(6)
        else:
            A_reg = A
        try:
            c_scaled = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            c_scaled = np.linalg.lstsq(A_reg, b, rcond=None)[0]

        pred = V @ c_scaled
        mse = np.mean((pred - y_smoothed) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_coeff_raw = c_scaled

    # Convert from scaled polynomial back to original x powers
    mu = x_mean
    sigma = x_std
    coeff = np.zeros(6)
    for k in range(6):
        for j in range(k, 6):
            binom_val = 1.0
            for m in range(k):
                binom_val *= (j - m) / (m + 1)
            coeff[k] += best_coeff_raw[j] * binom_val * ((-mu) ** (j - k)) / (sigma ** j)

    if not np.all(np.isfinite(coeff)):
        return np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
