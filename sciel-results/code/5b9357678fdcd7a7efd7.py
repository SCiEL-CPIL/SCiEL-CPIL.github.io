"""Direct analytical Bermudan put exercise policy via backward induction with
Gauss-Hermite quadrature and exact (unclipped) Gaussian transition.
Optimized with precomputed interpolation weights for speed, higher grid resolution,
and precise threshold refinement."""
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

    # Grid resolution - increased to 5000 for better precision
    n_grid = 5000
    S_min = 0.001 * K
    S_max = 8.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)

    # Gauss-Hermite quadrature
    # Increased to 350 nodes for better tail integration accuracy
    n_quad = 350
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)

    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    log_shifts = drift + vol_step * z_norm
    exp_shifts = np.exp(log_shifts)
    disc = np.exp(-r * dt)

    # Precompute interpolation structure for speed
    # For each quadrature node q, S_next[j, q] = S_grid[j] * exp_shifts[q]
    # We need to interpolate V_next at these points.
    # Since S_grid is geomspace, we can precompute the indices and weights.
    
    # indices[j, q] is the index i such that S_grid[i] <= S_grid[j]*exp_shifts[q] < S_grid[i+1]
    # weights[j, q] is the linear interpolation weight
    
    # We can compute this vectorized.
    # S_next is (n_grid, n_quad)
    S_next = S_grid[:, np.newaxis] * exp_shifts[np.newaxis, :]
    
    # Find indices
    # searchsorted returns index where S_next would be inserted to maintain order
    # We want the index of the lower bound
    # side='right' gives the first index where S_grid > S_next, so idx-1 is the lower bound
    idx_right = np.searchsorted(S_grid, S_next, side='right')
    idx_right = np.clip(idx_right, 1, n_grid - 1)
    
    # Lower index is idx_right - 1
    idx_lo = idx_right - 1
    idx_hi = idx_right
    
    # Get grid values
    S_lo = S_grid[idx_lo]
    S_hi = S_grid[idx_hi]
    
    # Calculate weights
    denom = S_hi - S_lo
    denom = np.where(denom > 0, denom, 1.0)
    w_interp = (S_next - S_lo) / denom
    w_interp = np.clip(w_interp, 0.0, 1.0)
    
    # Store these for reuse in the loop
    # We can construct a sparse matrix or just use the indices and weights directly
    # For speed, we will use the indices and weights to gather V_next values
    
    # Backward induction
    V_next = np.maximum(K - S_grid, 0.0)
    thresholds = np.zeros(n_dates)
    V_list = [None] * n_dates

    for t_idx in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 8.0:
            # Fallback to simple approximation if running out of time
            for remaining in range(t_idx, -1, -1):
                thresholds[remaining] = K * 0.85
            break

        # Continuation value calculation
        # V_interp[j, q] = V_next[idx_lo[j, q]] * (1 - w_interp[j, q]) + V_next[idx_hi[j, q]] * w_interp[j, q]
        
        V_lo = V_next[idx_lo]  # shape (n_grid, n_quad)
        V_hi = V_next[idx_hi]  # shape (n_grid, n_quad)
        
        V_interp = V_lo * (1.0 - w_interp) + V_hi * w_interp
        
        # Weighted sum over quadrature nodes
        # cont_value[j] = disc * sum_q V_interp[j, q] * w_norm[q]
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
            # Use a fixed number of iterations for speed, or time-based
            for _ in range(40):
                if time.time() > deadline - 8.0:
                    break
                mid = 0.5 * (lo + hi)
                # For refinement, we can use a simpler interpolation or the same precomputed structure
                # Since mid is a single value, we can just use np.interp which is fast for single points
                # Or we can reuse the vectorized logic but it's overkill for a single point
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
    # This is often where the most accuracy gain comes from
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
            width = 0.005 * K
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
                # No sign change, threshold might be at boundary or incorrect
                # Try expanding or just skip
                continue

            for _ in range(60):
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

    # Parametric smooth boundary fit
    if time.time() < deadline - 20.0:
        t_vals = np.arange(n_dates, dtype=float)
        t_norm = t_vals / max(n_dates - 1, 1)
        
        # Fit a quadratic in t_norm to log(threshold)
        log_thresh = np.log(np.maximum(thresholds, 1e-10))
        
        A = np.vstack([np.ones_like(t_norm), t_norm, t_norm**2]).T
        if len(log_thresh) >= 3:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_thresh, rcond=None)
            smooth_log_thresh = A @ coeffs
            smooth_thresholds = np.exp(smooth_log_thresh)
            smooth_thresholds = np.clip(smooth_thresholds, 0.0, K)
            
            # Only use smooth thresholds if they are close to the original
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
