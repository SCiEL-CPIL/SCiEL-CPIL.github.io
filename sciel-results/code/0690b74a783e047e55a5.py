"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolution_bound(x):
    """Compute the max |autoconvolution| for a signed sequence using FFT.
    Autoconvolution: circular convolution of x with itself."""
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.fft(x, size)
    # Autoconvolution in frequency domain: F * F (not F * conj(F) which is autocorrelation)
    ac = np.fft.ifft(F * F).real[:n]
    return np.max(np.abs(ac))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    seq = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        # Thue-Morse: parity of number of 1s in binary representation
        if bin(i).count('1') % 2 == 1:
            seq[i] = -1.0
    return seq

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill

    # Start with Thue-Morse sequence
    x = _thue_morse(N)
    best_x = x.copy()
    best_score = _autoconvolution_bound(best_x)

    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_x.astype(np.int8).tolist()

    # Local search: flip individual bits
    improved = True
    while improved:
        improved = False
        for i in range(N):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                return best_x.astype(np.int8).tolist()
            trial = best_x.copy()
            trial[i] = -trial[i]
            score = _autoconvolution_bound(trial)
            if score < best_score - 1e-12:
                best_score = score
                best_x = trial
                improved = True

    return best_x.astype(np.int8).tolist()
# EVOLVE-BLOCK-END
