"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _compute_c3(h):
    """
    Compute the autoconvolution upper bound C3 for a signed step function h.
    C3 is defined as the maximum absolute value of the autocorrelation at non-zero lags,
    normalized by the zero-lag value (which is N for binary +/-1 sequences).
    Lower is better.
    """
    # Compute autocorrelation via FFT
    # Pad to at least 2N to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0 (sum of squares = N)
    # r[1] to r[N-1] are positive lags
    # r[n_fft - 1] ... r[n_fft - (N-1)] are negative lags
    # For real signals, autocorrelation is symmetric, so we only need positive lags
    
    # The zero-lag value
    r0 = r[0]
    
    # Max absolute value at non-zero lags
    # Positive lags: r[1:N]
    # Negative lags are symmetric, so we don't need to check them separately
    max_sidelobe = np.max(np.abs(r[1:N]))
    
    # C3 is the ratio (upper bound on normalized autocorrelation peak)
    return max_sidelobe / r0


def _greedy_coordinate_descent(h, start_time, deadline):
    """
    Perform greedy coordinate descent: repeatedly flip the position that gives
    the largest improvement in C3 until no single flip improves the objective.
    """
    current_c3 = _compute_c3(h)
    improved = True
    
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_idx = -1
        best_c3 = current_c3
        
        # Try each position flip and keep the best improvement
        for idx in range(N):
            if (time.time() - start_time) > deadline - 10:
                break
            
            h[idx] *= -1
            c3_trial = _compute_c3(h)
            h[idx] *= -1  # revert
            
            if c3_trial < best_c3:
                best_c3 = c3_trial
                best_idx = idx
        
        # Apply the best flip if found
        if best_idx >= 0:
            h[best_idx] *= -1
            current_c3 = best_c3
            improved = True
    
    return h, current_c3


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Use multiple random initializations and greedy coordinate descent
    rng = np.random.default_rng(42)
    
    best_h = None
    best_c3 = float('inf')
    
    # Try multiple random starting points
    max_starts = 50
    for start_i in range(max_starts):
        if (time.time() - start_time) > deadline - 5:
            break
        
        # Random initial sequence
        h = rng.choice([-1, 1], size=N).astype(np.float64)
        
        # Run greedy coordinate descent from this starting point
        h_opt, c3_opt = _greedy_coordinate_descent(h, start_time, deadline)
        
        if c3_opt < best_c3:
            best_c3 = c3_opt
            best_h = h_opt.copy()
    
    # If we didn't find any valid result (shouldn't happen), return a random sequence
    if best_h is None:
        best_h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    return best_h
# EVOLVE-BLOCK-END
