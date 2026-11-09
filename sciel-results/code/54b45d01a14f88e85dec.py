"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _compute_autocorr_peak(h):
    """Compute max |autocorrelation| over non-zero lags using FFT."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # r[0] is DC (sum of squares = N for ±1)
    # Non-zero lags: r[1:N] and r[N+1:n_fft] (which mirror r[N-1:0:-1])
    # Due to symmetry, max over r[1:N] suffices
    return np.max(np.abs(r[1:N]))


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=N)
    
    best_peak = _compute_autocorr_peak(h)
    best_h = h.copy()
    
    # Local search: try flipping each bit, keep if it improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            # Try flipping bit i
            h[i] = -h[i]
            new_peak = _compute_autocorr_peak(h)
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
    
    # If we have time, do a second pass with random restarts
    if time.time() < deadline - 20.0:
        for seed in range(1, 10):
            if time.time() >= deadline - 5.0:
                break
            rng2 = np.random.default_rng(seed)
            h2 = rng2.choice([-1, 1], size=N)
            peak2 = _compute_autocorr_peak(h2)
            if peak2 < best_peak:
                best_peak = peak2
                best_h = h2.copy()
                # Local search from this new start
                for i in range(N):
                    if time.time() >= deadline - 5.0:
                        break
                    h2[i] = -h2[i]
                    new_peak = _compute_autocorr_peak(h2)
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_h = h2.copy()
                    else:
                        h2[i] = -h2[i]
    
    return best_h
# EVOLVE-BLOCK-END
