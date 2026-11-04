"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _autoconvolution_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding zero-lag self term)."""
    n = len(f)
    # Linear convolution via FFT
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    # conv[0] is the zero-lag (self) term: sum(f^2)
    # We want max |conv[k]| for k != 0
    if len(conv) > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0

def _compute_score(f):
    """The score is the upper bound on C3 (lower is better)."""
    return _autoconvolution_max_abs(f)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Initial best: all ones
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # Try some simple structured patterns first
    # Pattern 1: alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    score_alt = _compute_score(alt)
    if score_alt < best_score:
        best_score = score_alt
        best_f = alt.copy()
    
    # Pattern 2: blocks of 4
    block4 = np.ones(N)
    block4[1:4] = -1
    block4 = np.tile(block4, N//4)
    score_b4 = _compute_score(block4)
    if score_b4 < best_score:
        best_score = score_b4
        best_f = block4.copy()
    
    # Pattern 3: random signs
    np.random.seed(42)
    for trial in range(50):
        rand_f = np.random.choice([-1, 1], N)
        score_r = _compute_score(rand_f)
        if score_r < best_score:
            best_score = score_r
            best_f = rand_f.copy()
        if time.time() > deadline:
            break
    
    # Local search: flip individual elements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            old_val = best_f[i]
            best_f[i] = -old_val
            score_new = _compute_score(best_f)
            if score_new < best_score:
                best_score = score_new
                improved = True
            else:
                best_f[i] = old_val
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Pairwise swaps: flip two elements at once
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try random pairs
        np.random.seed(int(time.time()))
        for _ in range(1000):
            if time.time() > deadline:
                break
            i, j = np.random.choice(N, 2, replace=False)
            best_f[i] = -best_f[i]
            best_f[j] = -best_f[j]
            score_new = _compute_score(best_f)
            if score_new < best_score:
                best_score = score_new
                improved = True
            else:
                best_f[i] = -best_f[i]
                best_f[j] = -best_f[j]
        if time.time() > deadline:
            break
    
    return best_f
# EVOLVE-BLOCK-END
