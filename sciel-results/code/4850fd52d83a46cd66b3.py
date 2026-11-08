"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_sidelobe_max(h):
    """Compute max |R[k]| for k != 0 using FFT-based circular autocorrelation."""
    n = len(h)
    # Zero-pad to avoid circular aliasing for linear autocorrelation
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)[:n]
    # R[0] is the DC term (sum of squares = n for +/-1), sidelobes are R[1:] and R[-1:]
    # For linear autocorrelation of length n, lags are 0..n-1 and -(n-1)..-1
    # R from irfft gives circular, but with zero-padding it's linear for lags 0..n-1
    # Actually with nfft >= 2n-1, irfft gives linear autocorrelation for lags 0..nfft-1
    # R[0] = sum h^2, R[1..n-1] = positive lags, R[nfft-n+1..nfft-1] = negative lags
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft-n+1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = _autocorr_sidelobe_max(best_h)
    
    # Local search: try flipping each bit, keep if it improves
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > deadline - 5:
                break
            h[i] *= -1
            new_score = _autocorr_sidelobe_max(h)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[i] *= -1  # revert
    
    return best_h
# EVOLVE-BLOCK-END
