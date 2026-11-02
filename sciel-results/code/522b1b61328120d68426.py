import numpy as np
import time

N = 100
DEADLINE = 140  # seconds, safety margin under 150s limit

def _max_overlap_value(h):
    """Compute the maximum overlap of h with shifted 1-h over all circular shifts."""
    # O_k = sum_i h_i * (1 - h_{i+k}) for k = 1..N-1
    # We want max over k != 0
    # Use FFT-based circular cross-correlation
    # Overlap at shift k: sum_i h_i * (1 - h_{(i+k) mod N})
    # = N/2 - sum_i h_i * h_{(i+k) mod N}
    # sum_i h_i * h_{(i+k) mod N} is the circular autocorrelation at lag k
    
    # Compute autocorrelation via FFT
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    H = np.fft.rfft(h, n_fft)
    acf = np.fft.irfft(H * np.conj(H), n_fft)[:N]
    
    # acf[k] = sum_i h_i * h_{(i+k) mod N} for k=0..N-1
    # Overlap at shift k = N/2 - acf[k]
    # We want max over k=1..N-1
    overlaps = N / 2.0 - acf[1:]
    return float(np.max(overslaps) if len(overlaps) > 0 else 0.0)

def construct():
    t0 = time.time()
    
    # Start from constant sequence (known optimum candidate)
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    # Try to improve using structured perturbations + local search
    # The key insight: constant h=0.5 gives overlap = N/4 for all shifts.
    # We need to check if any non-constant h can do better.
    # From theory, constant is optimal, but let's verify with structured search.
    
    # Structured approach: try sequences with specific patterns
    # 1. Two-level sequences: a fraction p at value a, rest at value b
    #    Constraint: p*a + (1-p)*b = 0.5
    #    Try various p and a
    
    rng = np.random.default_rng(42)
    
    # Local search from constant
    lr = 0.05
    for iteration in range(2000):
        if time.time() - t0 > DEADLINE:
            break
        
        # Random perturbation in sum-zero direction
        d = rng.standard_normal(N)
        d -= np.mean(d)  # preserve sum
        d /= np.linalg.norm(d) * N  # scale
        
        h_try = h + lr * d
        h_try = np.clip(h_try, 0, 1)
        
        # Re-enforce sum constraint
        diff = N / 2.0 - np.sum(h_try)
        if abs(diff) > 1e-10:
            mask = (h_try > 1e-8) & (h_try < 1 - 1e-8)
            if np.sum(mask) > 0:
                h_try[mask] += diff / np.sum(mask)
                h_try = np.clip(h_try, 0, 1)
        
        val = _max_overlap_value(h_try)
        
        if val < best_val:
            best_val = val
            best_h = h_try.copy()
            h = h_try.copy()
            lr = min(lr * 1.05, 0.3)
        else:
            lr *= 0.95
    
    # Try a few structured initializations
    structures = [
        # Alternating high/low
        lambda i: 0.7 if i % 2 == 0 else 0.3,
        # Three-level
        lambda i: 0.8 if i % 3 == 0 else (0.3 if i % 3 == 1 else 0.4),
        # Sine-modulated
        lambda i: 0.5 + 0.2 * np.sin(2 * np.pi * i / N),
    ]
    
    for struct_fn in structures:
        if time.time() - t0 > DEADLINE:
            break
        
        h_s = np.array([struct_fn(i) for i in range(N)])
        # Normalize to sum N/2
        diff = N / 2.0 - np.sum(h_s)
        h_s += diff / N
        h_s = np.clip(h_s, 0, 1)
        
        val = _max_overlap_value(h_s)
        if val < best_val:
            best_val = val
            best_h = h_s.copy()
    
    # Final local search from best found
    h = best_h.copy()
    lr = 0.02
    for iteration in range(1000):
        if time.time() - t0 > DEADLINE:
            break
        
        d = rng.standard_normal(N)
        d -= np.mean(d)
        d /= max(np.linalg.norm(d) * N, 1e-10)
        
        h_try = h + lr * d
        h_try = np.clip(h_try, 0, 1)
        
        diff = N / 2.0 - np.sum(h_try)
        if abs(diff) > 1e-10:
            mask = (h_try > 1e-8) & (h_try < 1 - 1e-8)
            if np.sum(mask) > 0:
                h_try[mask] += diff / np.sum(mask)
                h_try = np.clip(h_try, 0, 1)
        
        val = _max_overlap_value(h_try)
        
        if val < best_val:
            best_val = val
            best_h = h_try.copy()
            h = h_try.copy()
            lr = min(lr * 1.03, 0.1)
        else:
            lr *= 0.97
    
    # Ensure valid output
    result = best_h.copy()
    diff = N / 2.0 - np.sum(result)
    if abs(diff) > 1e-9:
        mask = (result > 1e-8) & (result < 1 - 1e-8)
        if np.sum(mask) > 0:
            result[mask] += diff / np.sum(mask)
            result = np.clip(result, 0, 1)
    
    return result
