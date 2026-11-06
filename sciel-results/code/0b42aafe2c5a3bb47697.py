"""Rewritten program: degree-5 polynomial fit with robust IRLS + F-proxy variance model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 150.0  # wall-clock limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = x.size

    if n < 6:
        return np.zeros(6)

    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree-5 polynomial in scaled x
    # V[:, j] = x_scaled^j
    V = np.vander(x_scaled, N=6, increasing=True)

    # Initial OLS fit
    try:
        coeff_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return _rescale(coeff_scaled, x_mean, x_std)

    # Robust IRLS with Tukey's biweight
    # Compute initial residuals
    pred = V @ coeff_scaled
    resid = y - pred
    scale = np.median(np.abs(resid)) / 0.6745
    if scale < 1e-12:
        scale = np.std(resid) + 1e-12

    # Tukey's biweight weights
    def tukey_weights(r, c=4.685):
        ar = np.abs(r) / scale
        w = np.zeros_like(ar)
        mask = ar <= c
        w[mask] = (1.0 - (ar[mask] / c) ** 2) ** 2
        return w

    best_coeff = coeff_scaled.copy()
    best_score = np.inf

    for it in range(20):
        if time.time() > deadline - 10.0:
            break
        w = tukey_weights(resid)
        # Ensure at least some weight
        if w.sum() < n * 0.1:
            # Fallback to uniform
            w = np.ones(n)
        W = np.diag(w)
        try:
            coeff_scaled, _, _, _ = np.linalg.lstsq(V * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
        except np.linalg.LinAlgError:
            break
        pred = V @ coeff_scaled
        resid = y - pred
        new_scale = np.median(np.abs(resid)) / 0.6745
        if new_scale < 1e-12:
            new_scale = np.std(resid) + 1e-12
        scale = new_scale
        # Check convergence
        if np.max(np.abs(coeff_scaled - best_coeff)) < 1e-10:
            best_coeff = coeff_scaled.copy()
            break
        best_coeff = coeff_scaled.copy()

    # Now try to incorporate a variance-proxy: model the residual variance as a polynomial of x
    # This captures F-dependent scaling. We do a weighted fit where the weight is
    # inversely proportional to a local variance model.
    # Fit a degree-2 polynomial for local variance: var(x) = sum v_i * x_scaled^i
    # Use squared residuals from the current fit
    sq_resid = resid ** 2
    V2 = np.vander(x_scaled, N=3, increasing=True)
    try:
        var_coeff, _, _, _ = np.linalg.lstsq(V2, sq_resid, rcond=None)
        # Clip negative variance estimates
        var_pred = V2 @ var_coeff
        var_pred = np.maximum(var_pred, np.median(var_pred) * 0.1 + 1e-12)
        # Inverse-variance weights
        ivw = 1.0 / var_pred
        ivw /= np.sum(ivw) * n  # normalize so sum = 1
    except np.linalg.LinAlgError:
        ivw = None

    if ivw is not None:
        try:
            coeff_iv, _, _, _ = np.linalg.lstsq(V * np.sqrt(ivw)[:, None], y * np.sqrt(ivw), rcond=None)
            pred_iv = V @ coeff_iv
            mse_iv = np.mean((y - pred_iv) ** 2)
            mse_cur = np.mean(resid ** 2)
            if mse_iv < mse_cur:
                best_coeff = coeff_iv
        except np.linalg.LinAlgError:
            pass

    # Final high-precision polish: tight OLS on the best coefficient neighborhood
    try:
        # Use a very tight tolerance
        coeff_final, _, _, _ = np.linalg.lstsq(V, y, rcond=1e-15)
        pred_final = V @ coeff_final
        mse_final = np.mean((y - pred_final) ** 2)
        pred_best = V @ best_coeff
        mse_best = np.mean((y - pred_best) ** 2)
        if mse_final <= mse_best * 1.001:
            best_coeff = coeff_final
    except np.linalg.LinAlgError:
        pass

    return _rescale(best_coeff, x_mean, x_std)


def _rescale(coeff_scaled, x_mean, x_std):
    """Convert coefficients from scaled basis (x-mean)/std to original x basis."""
    # We have p(x_scaled) = sum c_i * x_scaled^i where x_scaled = (x - m)/s
    # We want q(x) = sum d_j * x^j such that q(x) = p((x-m)/s)
    # Use polynomial composition
    # p((x-m)/s) = sum c_i * ((x-m)/s)^i = sum c_i * s^{-i} * (x-m)^i
    # Expand (x-m)^i using binomial theorem
    n = len(coeff_scaled)
    d = np.zeros(n)
    for i in range(n):
        ci = coeff_scaled[i] / (x_std ** i)
        # (x - m)^i = sum_{j=0}^{i} C(i,j) * x^j * (-m)^{i-j}
        for j in range(i + 1):
            from math import comb
            d[j] += ci * comb(i, j) * ((-x_mean) ** (i - j))
    return d
# EVOLVE-BLOCK-END
