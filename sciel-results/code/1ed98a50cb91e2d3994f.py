"""Strategy: Orthogonal polynomial basis (Legendre) regression + Nadaraya-Watson denoising."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre


def _nadaraya_watson(x, y, x_eval, bandwidth):
    """Compute Nadaraya-Watson kernel regression estimate at x_eval points."""
    diff = x_eval[:, None] - x[None, :]
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
    wsum = np.sum(weights, axis=1)
    wsum[wsum < 1e-30] = 1e-30
    y_est = np.sum(weights * y[None, :], axis=1) / wsum
    return y_est


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()

    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]

    if len(x) == 0:
        return np.zeros(6)

    n = len(x)

    # Nadaraya-Watson denoising to smooth out stochastic fluctuations
    # Choose bandwidth based on data spread
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0

    # Try a few bandwidths and pick the one that gives best cross-validation on a subset
    if n > 200 and time.time() < DEADLINE - 10:
        # Use a subset for speed
        idx_cv = np.random.default_rng(42).choice(n, size=min(2000, n), replace=False)
        x_cv = x[idx_cv]
        t_cv = t[idx_cv]

        bandwidths = [0.01 * x_range, 0.02 * x_range, 0.05 * x_range, 0.1 * x_range]
        best_bw = 0.05 * x_range
        best_cv_mse = np.inf

        for bw in bandwidths:
            if time.time() > DEADLINE - 5:
                break
            # Leave-one-out style: split into two halves
            half = len(x_cv) // 2
            t_pred_1 = _nadaraya_watson(x_cv[half:], t_cv[half:], x_cv[:half], bw)
            t_pred_2 = _nadaraya_watson(x_cv[:half], t_cv[:half], x_cv[half:], bw)
            cv_mse = np.mean((t_pred_1 - t_cv[:half]) ** 2) + np.mean((t_pred_2 - t_cv[half:]) ** 2)
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_bw = bw

        # Apply NW smoothing to all data
        t_smoothed = _nadaraya_watson(x, t, x, best_bw)
    else:
        t_smoothed = t.copy()

    # Scale x to [-1, 1] for Legendre polynomials
    x_mid = (x_max + x_min) / 2.0
    x_half_range = x_range / 2.0
    if x_half_range < 1e-10:
        x_half_range = 1.0

    x_leg = (x - x_mid) / x_half_range

    # Build Legendre basis matrix: [P_0, P_1, ..., P_5]
    P = np.column_stack([eval_legendre(i, x_leg) for i in range(6)])

    # Ridge regression in orthogonal basis
    # Since Legendre polynomials are orthogonal, multicollinearity is eliminated
    # Use a small ridge for numerical stability
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0]

    # Train/validation split for alpha selection
    if n > 100:
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        split = int(0.8 * n)
        train_idx = idx[:split]
        val_idx = idx[split:]
        P_train = P[train_idx]
        t_train = t_smoothed[train_idx]
        P_val = P[val_idx]
        t_val = t_smoothed[val_idx]
    else:
        train_idx = np.arange(n)
        val_idx = train_idx
        P_train = P
        t_train = t_smoothed
        P_val = P
        t_val = t_smoothed

    best_coeff_leg = None
    best_val_mse = np.inf
    best_alpha = 0.0

    for alpha in alphas:
        if time.time() > DEADLINE:
            break
        if alpha == 0.0:
            coeff_leg, _, _, _ = np.linalg.lstsq(P_train, t_train, rcond=None)
        else:
            A = P_train.T @ P_train + alpha * np.eye(6)
            b = P_train.T @ t_train
            try:
                coeff_leg = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff_leg = np.linalg.lstsq(A, b, rcond=None)[0]

        pred_val = P_val @ coeff_leg
        val_mse = np.mean((pred_val - t_val) ** 2)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_coeff_leg = coeff_leg.copy()
            best_alpha = alpha

    if best_coeff_leg is None:
        # Fallback: linear fit
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, t, 1)[::-1]
        return coeff

    # Refit on all data with best alpha
    if best_alpha == 0.0:
        coeff_leg_all, _, _, _ = np.linalg.lstsq(P, t_smoothed, rcond=None)
    else:
        A_all = P.T @ P + best_alpha * np.eye(6)
        b_all = P.T @ t_smoothed
        try:
            coeff_leg_all = np.linalg.solve(A_all, b_all)
        except np.linalg.LinAlgError:
            coeff_leg_all = np.linalg.lstsq(A_all, b_all, rcond=None)[0]

    # Convert from Legendre basis to monomial basis
    # p(x_leg) = sum_i a_i * P_i(x_leg)
    # x_leg = (x - x_mid) / x_half_range
    # We need to express p(x_leg) as sum_j c_j * x^j
    #
    # P_0(z) = 1
    # P_1(z) = z
    # P_2(z) = (3z^2 - 1)/2
    # P_3(z) = (5z^3 - 3z)/2
    # P_4(z) = (35z^4 - 30z^2 + 3)/8
    # P_5(z) = (63z^5 - 70z^3 + 15z)/8
    #
    # z = (x - x_mid) / x_half_range
    #
    # Build the transformation matrix T such that P_i(z) = sum_j T[i,j] * z^j
    # Then z = (x - x_mid) / x_half_range, so z^j = (x - x_mid)^j / x_half_range^j
    # Then expand (x - x_mid)^j in powers of x

    # First, get coefficients in z (scaled) basis
    # P_i(z) expressed in monomial powers of z
    P_to_z = np.zeros((6, 6))
    P_to_z[0, 0] = 1.0
    P_to_z[1, 1] = 1.0
    P_to_z[2, 0] = -0.5
    P_to_z[2, 2] = 1.5
    P_to_z[3, 1] = -1.5
    P_to_z[3, 3] = 2.5
    P_to_z[4, 0] = 3.0 / 8.0
    P_to_z[4, 2] = -30.0 / 8.0
    P_to_z[4, 4] = 35.0 / 8.0
    P_to_z[5, 1] = 15.0 / 8.0
    P_to_z[5, 3] = -70.0 / 8.0
    P_to_z[5, 5] = 63.0 / 8.0

    # coeff in z monomial basis: d_j = sum_i a_i * P_to_z[i,j]
    d = P_to_z.T @ coeff_leg_all

    # Now z = (x - x_mid) / x_half_range
    # d_j * z^j = d_j * (x - x_mid)^j / x_half_range^j
    # (x - x_mid)^j = sum_{k=0}^{j} C(j,k) x^k * (-x_mid)^(j-k)

    coeff_orig = np.zeros(6)
    for j in range(6):
        c_j = d[j] / (x_half_range ** j)
        for k in range(j + 1):
            # C(j, k)
            if k == 0 or k == j:
                binom = 1.0
            else:
                num = 1
                for m in range(min(k, j - k)):
                    num *= (j - m)
                den = 1
                for m in range(1, min(k, j - k) + 1):
                    den *= m
                binom = num / den
            coeff_orig[k] += c_j * binom * ((-x_mid) ** (j - k))

    # Ensure finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff_orig
# EVOLVE-BLOCK-END
