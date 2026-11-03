"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    
    # Try different smooth shapes and pick the best
    # The normalized max autoconvolution is min >= 1, achieved at lag 0
    # We want the autoconvolution to be as flat as possible
    
    # Try a Gaussian-like shape
    center = (N - 1) / 2.0
    best_score = float('inf')
    best_h = None
    
    # Try various widths for Gaussian
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        if time.time() > deadline:
            break
        x = np.arange(N) - center
        h = np.exp(-(x ** 2) / (2.0 * (sigma * N) ** 2))
        # Ensure nonnegative
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            continue
        # Compute normalized max autoconvolution
        # Autoconvolution at lag k: sum_i h[i] * h[i+k]
        # Normalized by sum(h^2)
        l2_sq = np.sum(h ** 2)
        if l2_sq < 1e-15:
            continue
        # Compute autoconvolution using FFT for efficiency
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n_fft)
        conv = np.fft.irfft(F * F, n_fft)[:N]
        max_conv = np.max(conv)
        score = max_conv / l2_sq
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Also try raised cosine
    for width_frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if time.time() > deadline:
            break
        h = np.zeros(N)
        half_width = int(width_frac * N / 2.0)
        if half_width < 1:
            continue
        start = int(center) - half_width
        end = int(center) + half_width + 1
        start = max(0, start)
        end = min(N, end)
        if end <= start:
            continue
        x = np.arange(start, end)
        t = (x - center) / half_width
        h[start:end] = 0.5 * (1 + np.cos(np.pi * t))
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            continue
        l2_sq = np.sum(h ** 2)
        if l2_sq < 1e-15:
            continue
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n_fft)
        conv = np.fft.irfft(F * F, n_fft)[:N]
        max_conv = np.max(conv)
        score = max_conv / l2_sq
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        return [1.0] * N
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
