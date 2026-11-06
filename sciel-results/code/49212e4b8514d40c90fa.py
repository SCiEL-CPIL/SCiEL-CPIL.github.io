"""Rewritten program: degree-5 polynomial fit with robust IRLS + L2 regularization + variance model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

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

    # Regularization: exclude intercept (index 0) from penalty
    # P is diagonal: P[0] = 0, P[j] = 1 for j >= 1
    P = np.zeros(6)
    P[1:] = 1.0

    def solve_regularized(V_mat, y_vec, lam, P_diag=None):
        """Solve (V^T V + lam * P) c = V^T y with optional diagonal penalty."""
        if P_diag is None:
            P_diag = P
        A = V_mat.T @ V_mat + lam * np.diag(P_diag)
        b = V_mat.T @ y_vec
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    # Lambda grid for regularization strength
    # Scale relative to data magnitude
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0
    lam_grid = np.logspace(-8, 2, 17) * y_var / n

    # Initial OLS fit (no regularization)
    try:
        coeff_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return _rescale(np.zeros(6), x_mean, x_std)

    # Robust IRLS with Tukey's biweight
    pred = V @ coeff_scaled
    resid = y - pred
    scale = np.median(np.abs(resid)) / 0.6745
    if scale < 1e-12:
        scale = np.std(resid) + 1e-12

    def tukey_weights(r, c=4.685):
        ar = np.abs(r) / scale
        w = np.zeros_like(ar)
        mask = ar <= c
        w[mask] = (1.0 - (ar[mask] / c) ** 2) ** 2
        return w

    best_coeff = coeff_scaled.copy()
    best_mse = np.mean(resid ** 2)

    # IRLS iterations with regularization
    for it in range(15):
        if time.time() > deadline - 15.0:
            break
        w = tukey_weights(resid)
        if w.sum() < n * 0.1:
            w = np.ones(n)
        sqrt_w = np.sqrt(w)
        Vw = V * sqrt_w[:, None]
        yw = y * sqrt_w

        # Try each lambda value
        for lam in lam_grid:
            if time.time() > deadline - 15.0:
                break
            try:
                c_reg = solve_regularized(Vw, yw, lam)
                pred_reg = V @ c_reg
                mse_reg = np.mean((y - pred_reg) ** 2)
                if mse_reg < best_mse:
                    best_mse = mse_reg
                    best_coeff = c_reg
            except np.linalg.LinAlgError:
                continue

        # Update residuals with best coeff for next IRLS iteration
        pred = V @ best_coeff
        resid = y - pred
        new_scale = np.median(np.abs(resid)) / 0.6745
        if new_scale < 1e-12:
            new_scale = np.std(resid) + 1e-12
        scale = new_scale

        # Check convergence
        if it > 0 and np.max(np.abs(best_coeff - prev_coeff)) < 1e-10:
            break
        prev_coeff = best_coeff.copy()

    # Variance-proxy weighting: model residual variance as polynomial of x
    sq_resid = resid ** 2
    V2 = np.vander(x_scaled, N=3, increasing=True)
    ivw = None
    try:
        var_coeff, _, _, _ = np.linalg.lstsq(V2, sq_resid, rcond=None)
        var_pred = V2 @ var_coeff
        var_pred = np.maximum(var_pred, np.median(var_pred) * 0.1 + 1e-12)
        ivw = 1.0 / var_pred
        ivw /= np.sum(ivw) * n
    except np.linalg.LinAlgError:
        pass

    if ivw is not None:
        sqrt_ivw = np.sqrt(ivw)
        V_ivw = V * sqrt_ivw[:, None]
        y_ivw = y * sqrt_ivw
        for lam in lam_grid:
            if time.time() > deadline - 10.0:
                break
            try:
                c_iv = solve_regularized(V_ivw, y_ivw, lam)
                pred_iv = V @ c_iv
                mse_iv = np.mean((y - pred_iv) ** 2)
                if mse_iv < best_mse:
                    best_mse = mse_iv
                    best_coeff = c_iv
            except np.linalg.LinAlgError:
                continue

    # Final polish: try a few more lambda values with tight solve
    if time.time() < deadline - 5.0:
        for lam in [0.0] + list(lam_grid[:5]):
            try:
                c_final = solve_regularized(V, y, lam)
                pred_final = V @ c_final
                mse_final = np.mean((y - pred_final) ** 2)
                if mse_final < best_mse:
                    best_mse = mse_final
                    best_coeff = c_final
            except np.linalg.LinAlgError:
                continue

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
