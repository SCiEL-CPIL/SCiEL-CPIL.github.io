"""
Third autocorrelation inequality: choose signed step heights on 400 intervals
to minimize the autoconvolution upper bound C3.

Strategy:
- Represent the step function as a vector s of length 400 with values in {-1, +1}.
- The autoconvolution at lag 0 is sum(s^2) = N (fixed).
- The "upper bound on C3" is typically related to the maximum absolute value
  of the autoconvolution over all lags, or a norm thereof.
- We use a local search (sign-flip) approach with a time budget guard.
- Initialize with a pseudo-random sign pattern to avoid the trivial all-ones peak.
"""

import numpy as np
import time

N = 400
TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill

def _autoconv_peak(s):
    """
    Compute the maximum absolute value of the autoconvolution of s.
    For a step function with heights s[0..N-1], the autoconvolution at lag k
    is sum_{i} s[i] * s[i+k] (with appropriate bounds).
    We compute the full autoconvolution via FFT for efficiency.
    """
    # Zero-pad to avoid circular convolution artifacts
    size = 2 * N
    f = np.fft.rfft(s, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    # The autoconvolution is symmetric; we take the linear part
    # For a real signal, the linear autoconvolution at lag k (k=0..N-1) is:
    # ac_lin[k] = sum_{i=0}^{N-1-k} s[i] * s[i+k]
    # The FFT-based circular convolution gives us this if we zero-pad enough.
    # With size=2N, the linear convolution is embedded in ac[0:N] and ac[size-N+1:size]
    # Actually, for autoconvolution (correlation), we want:
    # r[k] = sum_{i=k}^{N-1} s[i] * s[i-k] for k >= 0
    # This is the same as the linear convolution of s with reversed s, or 
    # more directly, the inverse FFT of |F|^2 gives circular correlation.
    # With zero-padding to 2N, the circular correlation equals linear correlation.
    # r[k] for k=0..N-1 is in ac[0:N]
    # r[-k] = r[k] for real signals, so max |r| is max over ac[0:N]
    return np.max(np.abs(ac[0:N]))

def _try_flip(s, idx, current_peak):
    """
    Try flipping s[idx] and return the new peak if it improves.
    Uses incremental update of autoconvolution for speed.
    """
    old_val = s[idx]
    new_val = -old_val
    
    # Compute the change in autoconvolution
    # r[k] = sum_i s[i]*s[i+k]
    # Flipping s[idx] changes r[k] for all k where idx and idx+k are both in [0, N-1]
    # Specifically, for each k, the terms involving s[idx] are:
    #   s[idx] * s[idx+k] (when idx+k < N)
    #   s[idx-k] * s[idx] (when idx-k >= 0)
    # After flip, these become:
    #   -s[idx] * s[idx+k]
    #   s[idx-k] * (-s[idx])
    # So the change in r[k] is:
    #   -2 * s[idx] * s[idx+k] (for k where idx+k < N)
    #   -2 * s[idx-k] * s[idx] (for k where idx-k >= 0)
    # But note: when k=0, both terms refer to the same pair, so we need to be careful.
    
    # For simplicity and correctness, let's just recompute the peak after flip
    # using FFT. With N=400, this is fast enough.
    s[idx] = new_val
    new_peak = _autoconv_peak(s)
    s[idx] = old_val  # revert
    
    if new_peak < current_peak:
        s[idx] = new_val
        return new_peak
    return current_peak

def construct():
    start_time = time.time()
    
    # Initialize with a random sign pattern
    rng = np.random.default_rng(42)
    s = rng.choice([-1.0, 1.0], size=N)
    
    current_peak = _autoconv_peak(s)
    best_peak = current_peak
    best_s = s.copy()
    
    # Local search: try flipping each index, accept if it improves
    improved = True
    while improved and (time.time() - start_time) < TIME_LIMIT:
        improved = False
        for idx in range(N):
            if (time.time() - start_time) > TIME_LIMIT - 5:
                break
            old_peak = current_peak
            new_peak = _try_flip(s, idx, current_peak)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_s = s.copy()
    
    # Return the best configuration found
    return best_s

# Precompute nothing; construct() is the entry point
