"""Rewritten program for Erdos minimum overlap."""

import time
import numpy as np

N = 100

def _overlap_max(h):
    """Return the maximum of sum_i h_i*(1-h_{(i+k)%N}) over all k."""
    # Overlap for shift k: sum_i h_i * (1 - h_{(i+k)%N}) = sum h_i - sum h_i * h_{(i+k)%N}
    # sum h_i is constant (N/2), so minimizing max overlap = maximizing min of cross-correlation
    # But we want to MINIMIZE max overlap = MINIMIZE max(sum h_i - CC_k) = N/2 - MIN_CC
    # So minimizing max overlap is equivalent to maximizing the minimum cross-correlation.
    # Let's compute directly.
    h2 = 1.0 - h
    # Cross-correlation: for each k, sum_i h[i] * h2[(i+k) % N]
    # Use FFT for efficiency
    H = np.fft.rfft(h, n=2*N)
    H2 = np.fft.rfft(h2, n=2*N)
    # Circular cross-correlation
    cc = np.fft.irfft(H * np.conj(H2), n=2*N)[:N]
    # The overlap for shift k is sum_i h[i] * h2[(i+k)%N]
    # Wait, let me think about the indexing.
    # overlap_k = sum_i h_i * (1 - h_{(i+k)%N})
    # = sum h_i - sum h_i * h_{(i+k)%N}
    # sum h_i = N/2
    # sum h_i * h_{(i+k)%N} is the circular autocorrelation of h at lag k.
    # So overlap_k = N/2 - autocorr_k
    # To minimize max overlap_k, we want to maximize min autocorr_k.
    A = np.fft.rfft(h, n=2*N)
    auto = np.fft.irfft(A * np.conj(A), n=2*N)[:N]
    # auto[k] = sum_i h[i] * h[(i+k)%N] (approximately, up to scaling)
    # Actually, numpy's FFT gives: if x = ifft(fft(x)*conj(fft(x))), then x[k] = sum_i x[i]*x[(i-k)%N]
    # Let's just use direct computation for small N to be safe, or verify.
    # For N=100, direct is fast enough.
    max_overlap = 0.0
    for k in range(N):
        ov = np.sum(h * (1.0 - np.roll(h, k)))
        if ov > max_overlap:
            max_overlap = ov
    return max_overlap


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with baseline
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try a few parametric families
    # 1. Cosine: h_i = 0.5 + A * cos(2*pi*i/L + phase)
    # 2. Sine-cosine: h_i = 0.5 + A*cos(2*pi*i/L) + B*sin(2*pi*i/L)
    
    for L in range(2, N//2 + 1):
        if time.time() > deadline - 30.0:
            break
        for A in np.linspace(-0.4, 0.4, 9):
            for B in np.linspace(-0.4, 0.4, 9):
                if A == 0 and B == 0:
                    continue
                i = np.arange(N)
                h = 0.5 + A * np.cos(2 * np.pi * i / L) + B * np.sin(2 * np.pi * i / L)
                # Clip to [0, 1]
                h = np.clip(h, 0.0, 1.0)
                # Normalize sum to N/2
                current_sum = np.sum(h)
                if current_sum > 0:
                    h = h * (N / 2.0) / current_sum
                    # Re-clip after normalization
                    h = np.clip(h, 0.0, 1.0)
                    # Final normalization to ensure exact sum
                    current_sum = np.sum(h)
                    if current_sum > 0:
                        h = h * (N / 2.0) / current_sum
                score = _overlap_max(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Try random perturbations around best found
    best_h = best_h.copy()
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        h = best_h + np.random.normal(0, 0.05, N)
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        if current_sum > 0:
            h = h * (N / 2.0) / current_sum
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            if current_sum > 0:
                h = h * (N / 2.0) / current_sum
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final check
    if time.time() > deadline - 5.0:
        return best_h
    
    return best_h
