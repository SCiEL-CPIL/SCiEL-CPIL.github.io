"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _cross_correlations(h):
    """Compute circular cross-correlations of h with itself for all shifts."""
    n = len(h)
    # Linear cross-correlation via FFT, then fold to circular
    # r[k] = sum_i h[i] * h[(i+k) % n]
    # Using: circular corr = sum of linear corr at lags k and k-n
    hc = np.concatenate([h, h])
    # Cross-correlation: r[k] = sum_i h[i] * h[i+k] for k=0..n-1
    # Use FFT: IFFT(FFT(h) * conj(FFT(h))) gives autocorrelation
    H = np.fft.fft(h, n=2*n)
    ac = np.real(np.fft.ifft(H * np.conj(H)))[:n]
    return ac

def _max_overlap(h):
    """Compute max overlap of h with shifted (1-h)."""
    n = len(h)
    sum_h = np.sum(h)
    ac = _cross_correlations(h)
    # overlap[k] = sum_h - ac[k]
    overlaps = sum_h - ac
    return np.max(overslaps if False else overlaps)

def _objective(h):
    """The score: maximum overlap of h with shifted 1-h."""
    n = len(h)
    sum_h = np.sum(h)
    ac = _cross_correlations(h)
    overlaps = sum_h - ac
    return np.max(overlaps)

def construct():
    deadline = time.time() + 140.0
    n = N
    
    # Start with a binary pattern: first n/2 are 1, rest are 0
    # Then use simulated annealing to rearrange
    best_h = np.zeros(n)
    best_h[:n//2] = 1.0
    best_score = _objective(best_h)
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Simulated annealing
    T = 1.0
    cooling = 0.995
    
    # Precompute FFT of current h for fast updates? 
    # Actually, for binary swaps, we can update cross-correlations incrementally
    # But for simplicity and given N=100, full recompute is fast enough
    
    np.random.seed(42)
    
    while time.time() < deadline and T > 0.01:
        # Pick two random positions to swap
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        if current_h[i] == current_h[j]:
            continue
        
        # Swap
        current_h[i], current_h[j] = current_h[j], current_h[i]
        new_score = _objective(current_h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            # Revert
            current_h[i], current_h[j] = current_h[j], current_h[i]
        
        T *= cooling
    
    # Try to also explore non-binary solutions near the best binary solution
    # Relax: allow small perturbations
    h_relaxed = best_h.astype(float).copy()
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        idx = np.random.randint(0, n)
        old_val = h_relaxed[idx]
        new_val = np.clip(old_val + np.random.normal(0, 0.1), 0, 1)
        h_relaxed[idx] = new_val
        # Renormalize to keep sum = n/2
        s = np.sum(h_relaxed)
        if s > 0:
            h_relaxed = h_relaxed * (n / 2.0) / s
        h_relaxed = np.clip(h_relaxed, 0, 1)
        # Re-normalize again after clipping
        s2 = np.sum(h_relaxed)
        if s2 > 0:
            h_relaxed = h_relaxed * (n / 2.0) / s2
        score_rel = _objective(h_relaxed)
        if score_rel < best_score:
            best_score = score_rel
            best_h = h_relaxed.copy()
    
    # Final: ensure exact sum
    s = np.sum(best_h)
    if abs(s - n/2.0) > 1e-12:
        best_h = best_h * (n / 2.0) / s
    
    return best_h
# EVOLVE-BLOCK-END
