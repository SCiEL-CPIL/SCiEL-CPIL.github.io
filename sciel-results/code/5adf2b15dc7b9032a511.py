"""Direct analytical Bermudan put exercise policy via backward induction with
Gauss-Hermite quadrature and exact (unclipped) Gaussian transition.
Optimized with higher grid resolution, quadrature accuracy, and parametric
smooth boundary fitting to reduce discretization bias."""
# EVOLVE-BLOCK-START
import time
import numpy as np

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
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))

    if n_dates is None or n_dates <= 0:
        n_dates = 12

    dt = T / n_dates

    # Grid resolution - use log-spaced grid for better resolution near 0
    n_grid = 4000
    S_min = 0.002 * K
    S_max = 6.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)

    # Gauss-Hermite quadrature
    n_quad = 300
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)

    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    log_shifts = drift + vol_step * z_norm
    exp_shifts = np.exp(log_shifts)
    disc = np.exp(-r * dt)

    # Precompute interpolation indices for speed
    # For each grid point and quadrature node, find the interpolation interval
    # S_next[j, q] = S_grid[j] * exp_shifts[q]
    # We need to find i such that S_grid[i] <= S_next[j,q] < S_grid[i+1]
    
    # Backward induction
    V_next = np.maximum(K - S_grid, 0.0)
    thresholds = np.zeros(n_dates)
    V_list = [None] * n_dates

    for t_idx in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 8.0:
            for remaining in range(t_idx, -1, -1):
                thresholds[remaining] = K * 0.85
            break

        # Continuation value: disc * E[V_next(S * exp(log_shift))]
        # Vectorized computation
        S_next = S_grid[:, np.newaxis] * exp_shifts[np.newaxis, :]
        
        # Interpolate V_next at S_next
        # Use searchsorted for efficiency
        # For each column q, S_next[:, q] is sorted if S_grid is sorted
        # Since S_grid is geomspace (sorted), and exp_shifts[q] > 0, S_next[:, q] is sorted
        # Find indices for interpolation
        # S_next is shape (n_grid, n_quad)
        # For each q, find where S_next[:, q] falls in S_grid
        
        # Use np.searchsorted for each quadrature node
        # This is vectorized over grid points for each q
        indices = np.searchsorted(S_grid, S_next, side='right')
        # Clip to valid range
        indices = np.clip(indices, 1, n_grid - 1)
        
        # Get lower and upper grid points
        S_lo = S_grid[indices - 1]  # shape (n_grid, n_quad)
        S_hi = S_grid[indices]      # shape (n_grid, n_quad)
        
        # Interpolation weights
        denom = S_hi - S_lo
        denom = np.where(denom > 0, denom, 1.0)  # Avoid division by zero
        weight = (S_next - S_lo) / denom
        weight = np.clip(weight, 0.0, 1.0)
        
        # Interpolated values
        V_lo = V_next[indices - 1]  # shape (n_grid, n_quad)
        V_hi = V_next[indices]      # shape (n_grid, n_quad)
        V_interp = V_lo * (1.0 - weight) + V_hi * weight
        
        # Weighted sum over quadrature nodes
        cont_value = disc * (V_interp @ w_norm)

        immediate = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(immediate, cont_value)
        diff = immediate - cont_value

        # Find threshold via bisection on the difference
        # The exercise region is where diff >= 0, i.e., S < threshold
        pos_indices = np.where(diff >= 0)[0]
        if len(pos_indices) > 0:
            last_pos = pos_indices[-1]
            if last_pos + 1 < n_grid:
                d1 = diff[last_pos]
                d2 = diff[last_pos + 1]
                if d1 >= 0 and d2 < 0:
                    frac = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
                    thresh = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                else:
                    thresh = S_grid[last_pos]
            else:
                thresh = S_grid[last_pos]
        else:
            thresh = 0.0

        # Refine threshold by bisection
        thresh = np.clip(thresh, 0.0, K)
        if thresh > 0.0:
            lo = 0.0
            hi = thresh
            for _ in range(60):
                if time.time() > deadline - 8.0:
                    break
                mid = 0.5 * (lo + hi)
                S_m = mid * exp_shifts
                V_m = np.interp(S_m, S_grid, V_next)
                cont_m = disc * np.dot(V_m, w_norm)
                imm_m = max(K - mid, 0.0)
                d_m = imm_m - cont_m
                if d_m >= 0:
                    lo = mid
                else:
                    hi = mid
            thresh = 0.5 * (lo + hi)
            thresh = np.clip(thresh, 0.0, K)

        thresholds[t_idx] = thresh
        V_list[t_idx] = V_current
        V_next = V_current

    # Polish pass: refine thresholds using the correct future value functions
    if time.time() < deadline - 15.0:
        for t_idx in range(n_dates - 1, -1, -1):
            if time.time() > deadline - 8.0:
                break

            if t_idx + 1 < n_dates:
                V_next_polish = V_list[t_idx + 1]
            else:
                V_next_polish = np.maximum(K - S_grid, 0.0)

            thresh = thresholds[t_idx]
            if thresh <= 0.0:
                continue

            # Use a tighter bracket
            width = 0.01 * K
            lo = max(0.0, thresh - width)
            hi = min(K, thresh + width)

            # Evaluate at boundaries
            S_lo = lo * exp_shifts
            V_lo = np.interp(S_lo, S_grid, V_next_polish)
            cont_lo = disc * np.dot(V_lo, w_norm)
            imm_lo = max(K - lo, 0.0)
            d_lo = imm_lo - cont_lo

            S_hi = hi * exp_shifts
            V_hi = np.interp(S_hi, S_grid, V_next_polish)
            cont_hi = disc * np.dot(V_hi, w_norm)
            imm_hi = max(K - hi, 0.0)
            d_hi = imm_hi - cont_hi

            if d_lo * d_hi > 0:
                continue

            for _ in range(80):
                if time.time() > deadline - 8.0:
                    break
                mid = 0.5 * (lo + hi)
                S_m = mid * exp_shifts
                V_m = np.interp(S_m, S_grid, V_next_polish)
                cont_m = disc * np.dot(V_m, w_norm)
                imm_m = max(K - mid, 0.0)
                d_m = imm_m - cont_m
                if d_m >= 0:
                    lo = mid
                else:
                    hi = mid
            thresh = 0.5 * (lo + hi)
            thresh = np.clip(thresh, 0.0, K)
            thresholds[t_idx] = thresh

    # Parametric smooth boundary fit: fit a monotonic curve to the thresholds
    # to smooth out any discretization artifacts
    if time.time() < deadline - 20.0:
        # Fit a smooth monotonic curve to the thresholds
        t_vals = np.arange(n_dates, dtype=float)
        # Normalize t to [0,1]
        t_norm = t_vals / max(n_dates - 1, 1)
        
        # Fit a quadratic in t_norm to log(threshold) for smoothness
        log_thresh = np.log(np.maximum(thresholds, 1e-10))
        
        # Use least squares to fit log_thresh = c0 + c1*t_norm + c2*t_norm^2
        A = np.vstack([np.ones_like(t_norm), t_norm, t_norm**2]).T
        if len(log_thresh) >= 3:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_thresh, rcond=None)
            # Reconstruct smooth thresholds
            smooth_log_thresh = A @ coeffs
            smooth_thresholds = np.exp(smooth_log_thresh)
            smooth_thresholds = np.clip(smooth_thresholds, 0.0, K)
            
            # Only use smooth thresholds if they are close to the original
            # (i.e., the fit is good)
            if np.allclose(smooth_thresholds, thresholds, rtol=0.05, atol=0.5):
                thresholds = smooth_thresholds

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
