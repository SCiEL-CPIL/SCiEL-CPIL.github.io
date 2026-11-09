"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _autocorr_peak(h):
    """Compute max |autocorrelation| over non-zero lags using FFT."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # Non-zero lags: r[1:N]
    return np.max(np.abs(r[1:N]))


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    # Start with a random +/-1 sequence
    rng = np.random.default_rng(12345)
    h = rng.choice([-1, 1], size=N).astype(np.float64)

    best_peak = _autocorr_peak(h)
    best_h = h.copy()

    # Local search: iteratively flip bits to reduce peak autocorrelation
    n_iter = 0
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            h[i] *= -1  # flip
            peak = _autocorr_peak(h)
            if peak < best_peak:
                best_peak = peak
                best_h = h.copy()
                improved = True
            else:
                h[i] *= -1  # revert
            n_iter += 1
        if not improved:
            break

    return best_h
# EVOLVE-BLOCK-END
