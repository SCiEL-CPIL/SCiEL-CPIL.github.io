"""Combined closure: physics-informed polynomial fit with adaptive regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Build a shared degree-5 polynomial approximation to the unresolved
    fast-scale forcing of a two-scale Lorenz-96 model.

    Returns:
        coeff: array of 6 floats, coefficients c_0..c_5 for sum c_i * x**i
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Compute powers matrix
    powers = np.stack([x**i for i in range(6)], axis=1)

    # --- Approach 1: Ridge regression with adaptive lambda ---
    trace_term = np.trace(powers.T @ powers) / n
    if trace_term > 0:
        lam_base = 1e-4 * trace_term
    else:
        lam_base = 1e-10

    A = powers.T @ powers
    b_vec = powers.T @ y
    A_reg = A + lam_base * np.eye(6)

    try:
        coeff_ridge = np.linalg.solve(A_reg, b_vec)
    except np.linalg.LinAlgError:
        coeff_ridge = np.linalg.lstsq(powers, y, rcond=None)[0]

    if not np.all(np.isfinite(coeff_ridge)):
        coeff_ridge = np.zeros(6)

    elapsed = time.time() - start_time

    # --- Approach 2: Weighted least squares emphasizing central region ---
    coeff_wls = coeff_ridge.copy()
    if elapsed < DEADLINE - 10 and n >= 50:
        x_median = np.median(x)
        x_std = np.std(x)
        if x_std > 1e-12:
            z = (x - x_median) / x_std
            weights = np.exp(-0.5 * z**2)
            weights /= np.mean(weights)

            A_w = powers.T @ (weights[:, None] * powers)
            b_w = powers.T @ (weights * y)
            A_w_reg = A_w + lam_base * np.eye(6)
            try:
                coeff_wls = np.linalg.solve(A_w_reg, b_w)
                if not np.all(np.isfinite(coeff_wls)):
                    coeff_wls = coeff_ridge.copy()
            except np.linalg.LinAlgError:
                coeff_wls = coeff_ridge.copy()

    elapsed = time.time() - start_time

    # --- Approach 3: Cross-validation to select best lambda and method ---
    best_coeff = coeff_ridge.copy()
    best_mse = np.mean((powers @ coeff_ridge - y)**2)

    if elapsed < DEADLINE - 15 and n >= 200:
        # Generate candidate coefficients from different lambda values and methods
        candidates = [coeff_ridge, coeff_wls]

        # Try different lambda multipliers
        lambda_multipliers = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for lm in lambda_multipliers:
            if time.time() - start_time > DEADLINE - 10:
                break
            lam_trial = lm * (trace_term if trace_term > 0 else 1.0)
            A_trial = powers.T @ powers + lam_trial * np.eye(6)
            try:
                c_trial = np.linalg.solve(A_trial, b_vec)
                if np.all(np.isfinite(c_trial)):
                    candidates.append(c_trial)
            except np.linalg.LinAlgError:
                pass

            # Also try weighted version
            if x_std > 1e-12:
                A_w_trial = A_w + lam_trial * np.eye(6)
                try:
                    c_w_trial = np.linalg.solve(A_w_trial, b_w)
                    if np.all(np.isfinite(c_w_trial)):
                        candidates.append(c_w_trial)
                except np.linalg.LinAlgError:
                    pass

        # Evaluate each candidate using 2-fold cross-validation
        rng = np.random.RandomState(42)
        idx = np.arange(n)
        rng.shuffle(idx)
        fold_size = n // 2
        train_idx = idx[:fold_size]
        val_idx = idx[fold_size:]

        for c_cand in candidates:
            if time.time() - start_time > DEADLINE - 5:
                break
            if not np.all(np.isfinite(c_cand)):
                continue

            # Compute validation MSE
            pred_val = powers[val_idx] @ c_cand
            mse_val = np.mean((pred_val - y[val_idx])**2)

            if mse_val < best_mse:
                best_mse = mse_val
                best_coeff = c_cand

    # Final safety check
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff.astype(np.float64)
# EVOLVE-BLOCK-END
