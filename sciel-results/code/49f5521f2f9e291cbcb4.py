"""Climate dynamics: unresolved-tendency closure - robust polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-five polynomial to the unresolved subgrid tendency."""
    deadline = time.time() + 130  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    if n < 6:
        return np.zeros(6)

    # --- Step 1: Nadaraya-Watson denoising to estimate the underlying trend ---
    # Use a grid of bandwidths and select via leave-one-out cross-validation
    # (approximated by LOO-CV score for Gaussian kernel regression).
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0

    # Silverman's rule of thumb for initial bandwidth
    sigma = x.std()
    if sigma < 1e-12:
        sigma = x_range / 4.0
    h_silverman = 0.9 * min(sigma, x_range / 4.0) * n ** (-0.2)

    # Grid of bandwidths around Silverman
    h_grid = np.array([
        h_silverman * 0.3,
        h_silverman * 0.5,
        h_silverman * 0.7,
        h_silverman * 1.0,
        h_silverman * 1.5,
        h_silverman * 2.0,
        h_silverman * 3.0,
    ])
    h_grid = np.clip(h_grid, x_range * 0.005, x_range * 0.5)
    h_grid = np.unique(h_grid)

    # Sort x for efficient kernel evaluation
    x_sorted_idx = np.argsort(x)
    x_sorted = x[x_sorted_idx]
    y_sorted = y[x_sorted_idx]

    best_h = h_silverman
    best_cv_score = np.inf

    for h in h_grid:
        if time.time() > deadline - 20:
            break
        # Nadaraya-Watson estimate at all training points
        # K(u) = exp(-0.5 * u^2)
        # For efficiency, use the fact that x is 1D
        # Compute kernel weights for each point
        # This is O(n^2) which can be slow for large n
        # Use a binned approximation for large n
        if n > 2000:
            # Binned approach
            n_bins = min(200, n)
            bin_edges = np.linspace(x_min - h, x_max + h, n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = bin_edges[1] - bin_edges[0]
            y_bins = np.zeros(n_bins)
            counts = np.zeros(n_bins)
            bin_idx = np.clip(np.digitize(x_sorted, bin_edges) - 1, 0, n_bins - 1)
            np.add.at(y_bins, bin_idx, y_sorted)
            np.add.at(counts, bin_idx, 1)
            y_bins = np.where(counts > 0, y_bins / np.maximum(counts, 1), 0)

            # Interpolate smoothed values at training points
            y_smooth = np.interp(x_sorted, bin_centers, y_bins,
                                  left=y_bins[0], right=y_bins[-1])
        else:
            # Direct kernel regression
            y_smooth = np.zeros(n)
            # Process in chunks to manage memory
            chunk_size = min(n, 5000)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                x_chunk = x_sorted[start:end, np.newaxis]
                u = (x_chunk - x_sorted[np.newaxis, :]) / h
                K = np.exp(-0.5 * u * u)
                y_smooth[start:end] = (K @ y_sorted) / np.maximum(K.sum(axis=1), 1e-12)

        # LOO-CV score (MSE)
        residuals = y_sorted - y_smooth
        cv_score = np.mean(residuals ** 2)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_h = h

    # Final smoothed values with best bandwidth
    if n > 2000:
        n_bins = min(200, n)
        bin_edges = np.linspace(x_min - best_h, x_max + best_h, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_bins = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        bin_idx = np.clip(np.digitize(x_sorted, bin_edges) - 1, 0, n_bins - 1)
        np.add.at(y_bins, bin_idx, y_sorted)
        np.add.at(counts, bin_idx, 1)
        y_bins = np.where(counts > 0, y_bins / np.maximum(counts, 1), 0)
        y_smooth = np.interp(x_sorted, bin_centers, y_bins,
                              left=y_bins[0], right=y_bins[-1])
    else:
        y_smooth = np.zeros(n)
        chunk_size = min(n, 5000)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x_sorted[start:end, np.newaxis]
            u = (x_chunk - x_sorted[np.newaxis, :]) / best_h
            K = np.exp(-0.5 * u * u)
            y_smooth[start:end] = (K @ y_sorted) / np.maximum(K.sum(axis=1), 1e-12)

    # Map back to original order
    y_smooth_original = np.empty(n)
    y_smooth_original[x_sorted_idx] = y_smooth

    # --- Step 2: Initial polynomial fit (degree 5) ---
    # Use orthogonal polynomial basis for numerical stability
    # Shift and scale x to [-1, 1]
    x_mid = 0.5 * (x_min + x_max)
    x_scale = 0.5 * (x_max - x_min)
    if x_scale < 1e-12:
        x_scale = 1.0
    x_norm = (x - x_mid) / x_scale

    # Build Vandermonde matrix for degree 5
    V = np.vander(x_norm, 6, increasing=True)

    # Initial weighted least squares with Huber weights
    # Start with uniform weights
    w = np.ones(n)

    # Iterate reweighting (IRLS) with Huber loss
    # Huber delta: scale by MAD of residuals
    best_coeff_norm = None
    for iteration in range(20):
        if time.time() > deadline - 15:
            break

        # Weighted least squares
        W_sqrt = np.sqrt(w)
        Vw = V * W_sqrt[:, np.newaxis]
        yw = y_smooth * W_sqrt

        try:
            coeff_norm, residuals_ls, rank, sv = np.linalg.lstsq(Vw, yw, rcond=None)
        except np.linalg.LinAlgError:
            break

        if best_coeff_norm is None:
            best_coeff_norm = coeff_norm.copy()

        # Compute residuals in original space
        y_pred = V @ coeff_norm
        residuals = y_smooth - y_pred

        # Huber delta: 1.4826 * MAD
        mad = np.median(np.abs(residuals - np.median(residuals)))
        delta = 1.4826 * mad
        if delta < 1e-10:
            delta = np.std(residuals) * 0.6745  # fallback to ~1 sigma for Gaussian
        if delta < 1e-10:
            delta = 1.0

        # Update Huber weights
        abs_res = np.abs(residuals)
        w_new = np.where(abs_res <= delta, 1.0, delta / abs_res)
        w_new = np.clip(w_new, 0.01, 1.0)

        # Check convergence
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new

    if best_coeff_norm is None:
        # Fallback: unweighted least squares
        try:
            best_coeff_norm, _, _, _ = np.linalg.lstsq(V, y_smooth, rcond=None)
        except np.linalg.LinAlgError:
            return np.zeros(6)

    # --- Step 3: Convert from normalized basis to monomial basis ---
    # x_norm = (x - x_mid) / x_scale
    # We need coefficients c such that sum(c_i * x^i) = sum(a_i * x_norm^i)
    # x_norm = (x - x_mid) / x_scale = x/x_scale - x_mid/x_scale
    # Let alpha = 1/x_scale, beta = -x_mid/x_scale
    # x_norm = alpha * x + beta
    # x_norm^k = sum_{j=0}^{k} binom(k,j) * alpha^j * beta^(k-j) * x^j
    # So coefficient of x^j in the sum is:
    # c_j = sum_{k=j}^{5} a_k * binom(k,j) * alpha^j * beta^(k-j)

    alpha = 1.0 / x_scale
    beta = -x_mid / x_scale

    c = np.zeros(6)
    for k in range(6):
        a_k = best_coeff_norm[k]
        for j in range(k + 1):
            from math import comb
            c[j] += a_k * comb(k, j) * (alpha ** j) * (beta ** (k - j))

    # --- Step 4: Optional polish - refine with original (unsmoothed) data ---
    # Do a final IRLS fit using the raw y (not smoothed) with the same weights
    # This can help if smoothing introduced bias
    if time.time() < deadline - 10:
        # Recompute weights based on residuals from smoothed fit against raw data
        y_pred_raw = V @ best_coeff_norm
        residuals_raw = y - y_pred_raw
        mad_raw = np.median(np.abs(residuals_raw - np.median(residuals_raw)))
        delta_raw = 1.4826 * mad_raw
        if delta_raw < 1e-10:
            delta_raw = np.std(residuals_raw) * 0.6745
        if delta_raw < 1e-10:
            delta_raw = 1.0

        abs_res_raw = np.abs(residuals_raw)
        w_raw = np.where(abs_res_raw <= delta_raw, 1.0, delta_raw / abs_res_raw)
        w_raw = np.clip(w_raw, 0.01, 1.0)

        # One more IRLS iteration on raw data
        W_sqrt_raw = np.sqrt(w_raw)
        Vw_raw = V * W_sqrt_raw[:, np.newaxis]
        yw_raw = y * W_sqrt_raw
        try:
            coeff_norm_raw, _, _, _ = np.linalg.lstsq(Vw_raw, yw_raw, rcond=None)
        except np.linalg.LinAlgError:
            coeff_norm_raw = best_coeff_norm

        # Compare: use the one that gives lower weighted residual
        y_pred_smooth = V @ best_coeff_norm
        y_pred_raw = V @ coeff_norm_raw
        # Use normalized residual against smoothed data as proxy
        r_smooth = np.mean((y_smooth - y_pred_smooth) ** 2)
        r_raw = np.mean((y - y_pred_raw) ** 2)
        # Prefer the fit to smoothed data if it's better, otherwise raw
        # Actually, we want to minimize error on validation, so use smoothed fit
        # but check if raw fit is significantly better
        if r_raw < 0.9 * r_smooth:
            best_coeff_norm = coeff_norm_raw

        # Recompute monomial coefficients
        c = np.zeros(6)
        for k in range(6):
            a_k = best_coeff_norm[k]
            for j in range(k + 1):
                from math import comb
                c[j] += a_k * comb(k, j) * (alpha ** j) * (beta ** (k - j))

    # Ensure finite
    if not np.all(np.isfinite(c)):
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    return c
# EVOLVE-BLOCK-END
