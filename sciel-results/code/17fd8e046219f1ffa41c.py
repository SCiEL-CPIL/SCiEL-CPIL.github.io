"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def autoconv_peak(h):
        """Compute the peak (max absolute value) of the autoconvolution of h."""
        # Autoconvolution via FFT for efficiency
        n = len(h)
        # Use FFT to compute linear autoconvolution
        size = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * np.conj(H), size)[:2*n-1]
        return np.max(np.abs(ac))
    
    # Start with alternating signs
    h = np.ones(N)
    h[1::2] = -1.0
    
    # Try different patterns
    best_h = h.copy()
    best_peak = autoconv_peak(h)
    
    # Greedy local search: try flipping individual signs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            peak = autoconv_peak(h)
            if peak < best_peak:
                best_peak = peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = old_val
    
    # Try scaling: if we can scale down, the peak decreases
    # But there might be a constraint. Let's try small magnitudes.
    # Actually, if there's no constraint, we can make h -> 0 and peak -> 0.
    # So there must be a constraint. Let's assume L1 norm = 1 or sum of abs = N.
    
    # With |h_i| = 1, the peak at lag 0 is N. We can't avoid that.
    # So the metric must be different. Maybe it's the peak away from lag 0?
    # Or maybe it's normalized by the L1 norm squared?
    
    # Let's try: minimize max(|(f*f)[k]| / N) for k != 0, or the second peak.
    # Or maybe the task is to minimize the overall L-infinity of the autoconv
    # given that the function is a probability density (non-negative, integrates to 1).
    
    # Given the ambiguity, let's just return the alternating sign pattern
    # which at least has cancellation at odd lags.
    
    return best_h
# EVOLVE-BLOCK-END
