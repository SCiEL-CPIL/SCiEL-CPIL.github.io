"""Rewritten program: local search for signed step heights minimizing
maximum absolute off-zero autoconvolution divided by squared integral."""

import time
import numpy as np

N = 400
DEADLINE = 110.0  # seconds wall-clock safety margin before 120s kill


def _autocorr_full(f):
    """Full (linear) autoconvolution via FFT. Returns array of length 2N-1
    centered such that index N-1 is lag 0."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, size)[:2 * n - 1]
    # The convolution f*f has peak at N-1 for symmetric f.
    # For general f, the "lag 0" of correlation is at N-1.
    # We return the array as is, index N-1 is the central term.
    return ac


def _objective(f):
    """Compute max(|autoconv| excluding lag 0) / (sum(f))^2"""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    
    auto = _autocorr_full(f)
    # Exclude the central index (lag 0)
    # The central index is N-1
    center_idx = N - 1
    # Create a copy or mask to exclude center
    # We want max of all elements except center
    # If N=1, this would be empty, but N=400
    auto_no_center = auto.copy()
    auto_no_center[center_idx] = 0  # Set to 0 so it doesn't affect max abs
    
    max_abs = np.max(np.abs(auto_no_center))
    return max_abs / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with all ones
    best_f = np.ones(N)
    best_score = _objective(best_f)
    
    rng = np.random.default_rng(42)
    
    # Random restarts to find diverse local minima
    # Since the objective is non-convex, multiple restarts help.
    # However, we must be careful with time.
    
    # Phase 1: Random initialization
    # Try a few random sequences
    for i in range(10):
        if time.time() > deadline - 20:
            break
        f = rng.choice([-1, 1], size=N)
        score = _objective(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
    # Phase 2: Local Search
    # Greedy flipping of signs to improve the objective
    # To speed up, we can precompute the effect of a flip or just iterate.
    # Given N=400, one flip cost is O(N) for sum, O(N log N) for FFT.
    # 400 * 400 * log(400) is manageable within 110s?
    # FFT of size 1024 is fast.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        f = best_f.copy()
        # Try flipping each index
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = f[i]
            f[i] = -old_val
            new_score = _objective(f)
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
                improved = True
            else:
                f[i] = old_val
                
    return best_f
