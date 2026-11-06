"""Bermudan put exercise policy via analytical backward induction.

Uses a log-spaced value-function grid, exact (unclipped) Gaussian transition
evaluated with Gauss-Hermite quadrature, and direct root-finding on the smooth
immediate-minus-continuation function to locate each exercise threshold to
machine precision. A final local high-resolution polish refines each threshold
using the correct time-step-specific value function.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cont_at_scalar(S, S_grid, V_next, exp_shifts, w_norm, disc):
    """Continuation value at scalar S using the stored next-step value function."""
    S_next = S * exp_shifts
    V_interp = np.interp(S_next, S_grid, V_next)
    return disc * np.dot(V_interp, w_norm)


def _compute_cont_at_vector(S_vec, S_grid, V_next, exp_shifts, w_norm, disc):
    """Continuation value for vector of S values using the stored next-step value function."""
    S_next = S_vec[:, np.newaxis] * exp_shifts[np.newaxis, :]
    V_interp = np.interp(S_next.ravel(), S_grid, V_next).reshape(S_next.shape)
    return disc * (V_interp @ w_norm)


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using analytical backward induction.

    Args:
        training_paths: array of shape (n_paths, n_dates) with simulated spot prices
        contract: dict with contract parameters

    Returns:
        An object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    if n_dates is None or n_dates <= 0:
        n_dates = 12

    dt = T / n_dates

    # Value-function grid (log-spaced for resolution near 0)
    n_grid = 2500
    S_min = 0.005 * K
    S_max = 5.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)

    # Gauss-Hermite quadrature (exact Gaussian transition)
    n_quad = 300
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)

    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    exp_shifts = np.exp(drift + vol_step * z_norm)  # shape (n_quad,)
    disc = np.exp(-r * dt)

    # Backward induction
    V_next = np.maximum(K - S_grid, 0.0)

    thresholds = np.zeros(n_dates)
    # Store value functions for each time step for correct polish
    V_funcs = [None] * n_dates

    for t_idx in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 8.0:
            # Fallback heuristic for remaining dates
            for remaining in range(t_idx, -1, -1):
                thresholds[remaining] = K * 0.85
            break

        # Coarse scan to bracket the zero of f (f decreasing in S)
        scan = np.geomspace(S_min, min(K, S_max), 300)
        # Vectorized computation of continuation values for all scan points
        cont_scan = _compute_cont_at_vector(scan, S_grid, V_next, exp_shifts, w_norm, disc)
        imm_scan = np.maximum(K - scan, 0.0)
        f_scan = imm_scan - cont_scan

        # Find last index where f >= 0 (exercise region is S < threshold)
        pos_idx = np.where(f_scan >= 0.0)[0]
        if len(pos_idx) == 0:
            thresh = 0.0
        else:
            last_pos = pos_idx[-1]
            if last_pos + 1 < len(scan) and f_scan[last_pos + 1] < 0.0:
                # Linear interp to bracket
                d1 = f_scan[last_pos]
                d2 = f_scan[last_pos + 1]
                frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                lo = scan[last_pos]
                hi = scan[last_pos + 1]
            else:
                # f positive everywhere scanned -> threshold beyond scan
                lo = scan[-1]
                hi = min(K, S_max)

            # Bisection on the smooth f to high precision
            lo = max(0.0, lo)
            hi = min(K, hi)
            if hi > lo:
                for _ in range(50):
                    if time.time() > deadline - 8.0:
                        break
                    mid = 0.5 * (lo + hi)
                    fm = _compute_cont_at_scalar(mid, S_grid, V_next, exp_shifts, w_norm, disc)
                    imm = max(K - mid, 0.0)
                    fm = imm - fm
                    if fm >= 0.0:
                        lo = mid
                    else:
                        hi = mid
                thresh = 0.5 * (lo + hi)
            else:
                thresh = lo

        thresh = np.clip(thresh, 0.0, K)
        thresholds[t_idx] = thresh
        V_funcs[t_idx] = V_next.copy()

        # Update value function for next step back
        S_next_grid = S_grid[:, np.newaxis] * exp_shifts[np.newaxis, :]
        V_interp = np.interp(S_next_grid.ravel(), S_grid, V_next).reshape(S_next_grid.shape)
        cont_value = disc * (V_interp @ w_norm)
        immediate = np.maximum(K - S_grid, 0.0)
        V_next = np.maximum(immediate, cont_value)

    # Final high-resolution local polish on each threshold using correct V_func
    if time.time() < deadline - 10.0:
        for t_idx in range(n_dates):
            if time.time() > deadline - 5.0:
                break
            thresh = thresholds[t_idx]
            if thresh <= 0.0:
                continue
            V_func_t = V_funcs[t_idx]
            if V_func_t is None:
                continue
            local_width = max(0.003 * K, 0.02)
            local_lo = max(0.0, thresh - local_width)
            local_hi = min(K, thresh + local_width)
            if local_hi <= local_lo:
                continue
            n_local = 400
            S_local = np.linspace(local_lo, local_hi, n_local)
            cont_local = _compute_cont_at_vector(S_local, S_grid, V_func_t, exp_shifts, w_norm, disc)
            imm_local = np.maximum(K - S_local, 0.0)
            diff_local = imm_local - cont_local

            pos_idx = np.where(diff_local >= 0.0)[0]
            if len(pos_idx) > 0:
                last_pos = pos_idx[-1]
                if last_pos + 1 < n_local and diff_local[last_pos + 1] < 0.0:
                    d1 = diff_local[last_pos]
                    d2 = diff_local[last_pos + 1]
                    frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                    lo_f = S_local[last_pos]
                    hi_f = S_local[last_pos + 1]
                else:
                    lo_f = max(0.0, thresh - local_width)
                    hi_f = S_local[last_pos]
            else:
                lo_f = local_lo
                hi_f = local_lo

            if hi_f > lo_f:
                for _ in range(40):
                    if time.time() > deadline - 5.0:
                        break
                    mid_f = 0.5 * (lo_f + hi_f)
                    cont_f = _compute_cont_at_scalar(mid_f, S_grid, V_func_t, exp_shifts, w_norm, disc)
                    imm_f = max(K - mid_f, 0.0)
                    fm_f = imm_f - cont_f
                    if fm_f >= 0.0:
                        lo_f = mid_f
                    else:
                        hi_f = mid_f
                thresh_fine = 0.5 * (lo_f + hi_f)
                thresh_fine = np.clip(thresh_fine, 0.0, K)
            else:
                thresh_fine = lo_f
            thresholds[t_idx] = thresh_fine

    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time_idx):
            current_state = np.asarray(current_state)
            if time_idx < 0 or time_idx >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return current_state < threshold

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
