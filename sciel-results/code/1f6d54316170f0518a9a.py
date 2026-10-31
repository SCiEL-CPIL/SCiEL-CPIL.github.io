"""Structural change: full-dimensional simulated annealing on the 600-bin height vector,
with FFT-based convolution, mass-redistribution perturbations, and a cooling schedule.
This escapes the uniform critical point by allowing asymmetric shapes to form."""
import numpy as np
import time

N = 600

def _autoconv_peak_fft(h):
    """Compute max of linear autoconvolution using FFT."""
    # Pad to next power of 2 for efficiency
    n = 2 * N
    size = 1
    while size < n:
        size *= 2
    H = np.fft.rfft(h, size)
    C = np.fft.irfft(H * H, size)
    # Linear convolution is in C[0:N+N-1]
    return C[:2 * N - 1].max()

def _normalized_peak(h):
    s = h.sum()
    if s <= 0:
        return float('inf')
    return _autoconv_peak_fft(h) / (s * s)

def construct():
    start = time.time()
    deadline = start + 115.0  # 5s safety margin

    # Initialize: uniform (the seed)
    h = np.ones(N)
    best_h = h.copy()
    best_score = _normalized_peak(best_h)

    # Simulated annealing parameters
    T = 0.01  # initial temperature (relative to score scale)
    T_min = 1e-6
    cooling_rate = 0.9995  # geometric cooling
    # Each "step" is a mass redistribution between 2 random bins

    rng = np.random.default_rng(42)

    while time.time() < deadline:
        # Generate perturbation: move mass from one bin to another
        i = rng.integers(N)
        j = rng.integers(N)
        if i == j:
            continue

        # Move a fraction of h[j] to h[i]
        # The fraction is controlled by temperature
        frac = min(0.5, T * 50)  # larger T -> larger moves
        delta = h[j] * frac * (1 if rng.random() < 0.5 else -1)

        new_h = h.copy()
        new_h[i] += delta
        new_h[j] -= delta

        # Enforce nonnegativity
        if new_h[i] < 0 or new_h[j] < 0:
            # Clip and adjust
            if new_h[i] < 0:
                delta += new_h[i]
                new_h[i] = 0
                new_h[j] = h[j] - delta
            if new_h[j] < 0:
                new_h[j] = 0
                new_h[i] = h[i] + h[j] - new_h[j]
        
        new_score = _normalized_peak(new_h)
        old_score = _normalized_peak(h)

        # Accept if better, or with probability exp(-delta/T) if worse
        if new_score < old_score:
            h = new_h
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            delta_score = new_score - old_score
            if T > 0 and rng.random() < np.exp(-delta_score / (T + 1e-15)):
                h = new_h

        # Cool down
        T *= cooling_rate

        # Periodic restart: if stuck, try a larger random perturbation
        # Check every ~500 iterations by tracking time
        if (time.time() - start) % 2.0 < 0.01:  # roughly every 2 seconds
            # Try a few random asymmetric perturbations
            for _ in range(10):
                candidate = best_h.copy()
                # Add a random asymmetric component
                noise = rng.normal(0, 0.1, N)
                candidate += noise
                candidate = np.maximum(candidate, 0)
                if candidate.sum() > 0:
                    cand_score = _normalized_peak(candidate)
                    if cand_score < best_score:
                        best_score = cand_score
                        best_h = candidate.copy()
                        h = candidate.copy()
                        T = max(T, 0.005)  # reheat slightly

    # Final: ensure nonnegativity and return
    best_h = np.maximum(best_h, 0)
    return list(best_h)
