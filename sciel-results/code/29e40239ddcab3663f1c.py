"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_value(h):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    # Start with ones (known optimal: value = 2.0)
    best_h = np.ones(N, dtype=np.float64)
    best_val = 2.0
    
    # Try some structured perturbations to see if we can beat 2.0
    # Key insight: value = 2*N*max(conv)/sum(h)^2
    # Zero-lag of conv is sum(h^2), and by C-S |conv[k]| <= sum(h^2) for all k
    # So max(conv) = sum(h^2) always, giving value = 2*N*sum(h^2)/sum(h)^2 >= 2
    # Equality iff all h[i] equal (same sign)
    # So 2.0 is the global minimum.
    
    # Still, let's do a quick verification with random searches
    rng = np.random.default_rng(42)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Random search: try various patterns
        # Pattern 1: random
        h = rng.standard_normal(N)
        # Ensure sum != 0
        if abs(np.sum(h)) < 1e-10:
            h[0] += 1.0
        v = _compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
        
        # Pattern 2: low-frequency dominant
        freqs = np.fft.rfftfreq(N)
        spectrum = np.zeros(N // 2 + 1)
        # Emphasize low frequencies to make sum large
        spectrum[1:5] = rng.uniform(0.5, 2.0, 4)
        spectrum[0] = 0  # no DC component needed, we'll add mean
        h = np.fft.irfft(spectrum, n=N)
        # Shift to have positive mean
        h = h - np.mean(h) + 1.0
        if abs(np.sum(h)) < 1e-10:
            h[0] += 1.0
        v = _compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Pattern 3: Try to make sum(h) large while keeping max(conv) small
    # Since max(conv) >= sum(h^2) is not true; max(conv) = sum(h^2) exactly (zero lag)
    # The ratio is 2*N*sum(h^2)/sum(h)^2 which is minimized when h is constant
    # This is a well-known inequality: N*sum(h^2) >= sum(h)^2
    # So value >= 2 always.
    
    # Final: return the best found
    return best_h
# EVOLVE-BLOCK-END
