"""Improved closure: Robust polynomial fit using Iteratively Reweighted Least Squares (IRLS)
with Huber loss to mitigate noise/outliers, followed by basis conversion. Time-gated."""
import time
import numpy as np

_EARLY_DEADLINE_FRACTION = 0.85  # Safety margin: stop at 102s of 120s


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    x = np.asarray(training_x, dtype=np.float64).ravel()
    t = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = x.shape[0]
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n != t.shape[0]:
        m = min(n, t.shape[0])
        x = x[:m]
        t = t[:m]
        n = m
        if n == 0:
            return np.zeros(6, dtype=np.float64)

    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= 1e-14:
        c = np.zeros(6, dtype=np.float64)
        c[0] = float(np.mean(t))
        return c

    u = (x - mu) / sigma
    t_mean = float(np.mean(t))
    t_std = float(np.std(t))
    if t_std <= 1e-14:
        t_std = 1.0
    t_hat = (t - t_mean) / t_std

    V = np.vander(u, N=6, increasing=True)

    # Initial unweighted least squares fit to estimate scale of residuals
    try:
        w_init, *_ = np.linalg.lstsq(V, t_hat, rcond=None)
    except np.linalg.LinAlgError:
        w_init = np.zeros(6, dtype=np.float64)

    residuals = t_hat - V @ w_init
    # Estimate robust scale (MAD) to set Huber delta
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-10:
        mad = np.std(residuals)
    if mad < 1e-10:
        mad = 1.0
    delta = 1.4826 * mad

    # IRLS loop
    weights = np.ones(n, dtype=np.float64)
    best_w = w_init
    max_iter = 10
    tol = 1e-4

    for i in range(max_iter):
        if time.time() - t0 > _EARLY_DEADLINE_FRACTION * 120.0:
            break
            
        # Weighted Least Squares: min sum w_i * (y_i - V_i @ w)^2
        # This is equivalent to (V^T W V) w = V^T W y
        # Use Cholesky or Solve for stability
        W = np.diag(weights)
        try:
            # Efficient: (V * weights[:, None]).T @ V
            VW = V.T * weights
            ATA = VW @ V
            ATy = VW @ t_hat
            # Add tiny ridge for numerical stability if needed, though IRLS usually handles it
            # We stick to pure WLS here to let the weights do the regularization
            w_new = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            try:
                w_new, *_ = np.linalg.lstsq(V * np.sqrt(weights)[:, None], t_hat * np.sqrt(weights), rcond=None)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(w_new)):
            break

        # Update weights based on Huber loss
        residuals = t_hat - V @ w_new
        abs_res = np.abs(residuals)
        new_weights = np.where(abs_res <= delta, 1.0, delta / abs_res)
        
        # Check convergence
        if np.max(np.abs(new_weights - weights)) < tol:
            weights = new_weights
            best_w = w_new
            break
            
        weights = new_weights
        best_w = w_new

    # Fallback if IRLS failed or didn't run
    if best_w is None or not np.all(np.isfinite(best_w)):
        try:
            best_w, *_ = np.linalg.lstsq(V, t_hat, rcond=None)
        except np.linalg.LinAlgError:
            best_w = np.zeros(6, dtype=np.float64)

    # Convert standardized-basis coefficients w to raw-basis coefficients c
    from math import comb
    M2 = np.zeros((6, 6), dtype=np.float64)
    for j in range(6):
        for k in range(j, 6):
            M2[j, k] = comb(k, j) * (sigma ** j) * (mu ** (k - j))
    
    target = t_std * best_w
    try:
        c_dev = np.linalg.solve(M2, target)
    except np.linalg.LinAlgError:
        c_dev, *_ = np.linalg.lstsq(M2, target, rcond=None)

    c = c_dev.copy()
    c[0] += t_mean

    c = np.where(np.isfinite(c), c, 0.0)
    return c
