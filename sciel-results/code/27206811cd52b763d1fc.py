"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse ±1 sequence of length n."""
    seq = np.zeros(n, dtype=np.int8)
    seq[0] = 1
    length = 1
    while length < n:
        # Thue-Morse: append negation of current sequence
        seq[length:2*length] = -seq[:length]
        length *= 2
    return seq[:n]


def _autocorr_peak(h):
    """Compute max absolute autocorrelation (excluding lag 0) using FFT."""
    n = len(h)
    # Use FFT to compute full autocorrelation
    f = np.fft.fft(h.astype(np.float64), n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # acf[0] is the energy (sum of squares), we want max |acf[k]| for k>=1
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0


def _local_search(h, max_time, seed=42):
    """Local search: flip bits to reduce autocorrelation peak."""
    rng = np.random.RandomState(seed)
    h = h.copy().astype(np.int8)
    n = len(h)
    best_peak = _autocorr_peak(h)
    best_h = h.copy()
    deadline = time.time() + max_time
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try random flips
        improved = False
        for _ in range(50):  # try up to 50 random flips per iteration
            if time.time() >= deadline:
                break
            idx = rng.randint(n)
            h[idx] *= -1
            peak = _autocorr_peak(h)
            if peak < best_peak:
                best_peak = peak
                best_h = h.copy()
                no_improve_count = 0
                improved = True
            else:
                h[idx] *= -1  # revert
        if not improved:
            no_improve_count += 50
    
    return best_h, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Start with Thue-Morse sequence
    h = _thue_morse(N)
    best_h = h.copy()
    best_peak = _autocorr_peak(h)
    
    # Local search with deadline guard
    remaining = deadline - time.time()
    if remaining > 5:  # need at least 5 seconds for search
        h2, peak2 = _local_search(h, remaining - 5, seed=42)
        if peak2 < best_peak:
            best_h = h2
            best_peak = peak2
        
        # Second round with different seed if time permits
        remaining = deadline - time.time()
        if remaining > 10:
            h3, peak3 = _local_search(best_h, remaining - 10, seed=123)
            if peak3 < best_peak:
                best_h = h3
                best_peak = peak3
    
    return best_h
# EVOLVE-BLOCK-END

