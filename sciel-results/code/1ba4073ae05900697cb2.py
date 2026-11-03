"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Model the step function as binary on/off intervals and use simulated
annealing to stochastically flip individual intervals, minimizing the peak
autoconvolution. The discrete nature of the 600-bin structure is leveraged
to escape local minima that frustrate continuous gradient-based methods.
"""

import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 145  # safety margin from 150s

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fast_peak(f):
    """Fast computation of peak autoconvolution / (sum(f))^2 using FFT."""
    f = np.asarray(f, dtype=np.float64)
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def simulated_annealing(f_init, start_time, max_time):
    """Simulated annealing to minimize peak autoconvolution of binary step function."""
    f = f_init.copy().astype(np.float64)
    # Make it binary: threshold at 0.5
    f[f > 0.5] = 1.0
    f[f <= 0.5] = 0.0
    
    current_score = fast_peak(f)
    best_f = f.copy()
    best_score = current_score
    
    # Annealing schedule
    T_start = 0.1
    T_end = 1e-6
    # We'll adapt the number of iterations based on time
    
    rng = np.random.RandomState(12345)
    
    # Precompute indices
    indices = np.arange(N)
    
    # For efficiency, track which positions are on
    on_mask = f > 0.5
    n_on = np.sum(on_mask)
    
    # Adaptive step: try single flips first, then occasional multi-flips
    t = start_time
    elapsed = t - start_time
    max_elapsed = max_time - start_time
    
    # Determine number of iterations we can afford
    # Each iteration: one flip + score computation
    # Score computation via fftconvolve is ~O(N log N)
    # Let's estimate: ~0.5ms per iteration, so ~20000 iterations in 10s
    # But we need to be conservative
    
    while time.time() - start_time < max_time - 5:
        # Annealing temperature
        progress = (time.time() - start_time - start_time) / max_elapsed if max_elapsed > 0 else 0
        # Better: use a fixed number of iterations and track progress
        T = T_start * (T_end / T_start) ** ((time.time() - start_time - start_time) / max_elapsed)
        if T < T_end:
            T = T_end
        
        # Choose flip strategy
        # 80% single flip, 15% double flip, 5% block flip
        r = rng.random()
        if r < 0.80:
            # Single flip
            idx = rng.randint(N)
            old_val = f[idx]
            f[idx] = 1.0 - old_val
            
            new_score = fast_peak(f)
            
            # Accept or reject
            if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[idx] = old_val  # revert
        elif r < 0.95:
            # Double flip: flip two positions
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            old1, old2 = f[idx1], f[idx2]
            f[idx1] = 1.0 - old1
            f[idx2] = 1.0 - old2
            
            new_score = fast_peak(f)
            
            if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[idx1] = old1
                f[idx2] = old2
        else:
            # Block flip: flip a contiguous block of 3-7 positions
            block_size = rng.randint(3, 8)
            start_idx = rng.randint(N - block_size + 1)
            end_idx = start_idx + block_size
            old_block = f[start_idx:end_idx].copy()
            f[start_idx:end_idx] = 1.0 - f[start_idx:end_idx]
            
            new_score = fast_peak(f)
            
            if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[start_idx:end_idx] = old_block
    
    return best_f, best_score

def local_polish(f, start_time, max_time):
    """Local polish: try all single-bit flips to find improvements."""
    f = f.copy().astype(np.float64)
    f[f > 0.5] = 1.0
    f[f <= 0.5] = 0.0
    
    current_score = fast_peak(f)
    best_f = f.copy()
    best_score = current_score
    improved = True
    
    while improved and time.time() - start_time < max_time - 3:
        improved = False
        for idx in range(N):
            if time.time() - start_time > max_time - 3:
                break
            old_val = f[idx]
            f[idx] = 1.0 - old_val
            new_score = fast_peak(f)
            
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
                current_score = new_score
                improved = True
            else:
                f[idx] = old_val  # revert
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Multiple initializations for simulated annealing
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(N))
    
    # 2. Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    initializations.append(alt)
    
    # 3. Random sparse (30% on)
    rng = np.random.RandomState(42)
    sparse = (rng.random(N) < 0.3).astype(np.float64)
    initializations.append(sparse)
    
    # 4. Random dense (70% on)
    dense = (rng.random(N) < 0.7).astype(np.float64)
    initializations.append(dense)
    
    # 5. Block structure: alternating blocks of 10 on/off
    blocks = np.zeros(N)
    for i in range(0, N, 20):
        blocks[i:i+10] = 1.0
    initializations.append(blocks)
    
    # 6. Random medium (50% on)
    medium = (rng.random(N) < 0.5).astype(np.float64)
    initializations.append(medium)
    
    # 7. Concentrated center
    center = np.zeros(N)
    center[N//4:3*N//4] = 1.0
    initializations.append(center)
    
    # 8. Two peaks
    two_peaks = np.zeros(N)
    two_peaks[N//8:3*N//8] = 1.0
    two_peaks[5*N//8:7*N//8] = 1.0
    initializations.append(two_peaks)
    
    best_f = None
    best_score = np.inf
    
    # Run simulated annealing from each initialization
    n_init = len(initializations)
    time_per_init = (TIME_LIMIT - 10) / n_init  # leave 10s for final polish
    
    for i, init in enumerate(initializations):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        remaining = TIME_LIMIT - 10 - (time.time() - start_time)
        alloc = min(time_per_init, remaining)
        
        if alloc < 5:
            break
        
        f_sa, score_sa = simulated_annealing(init, start_time, start_time + alloc)
        
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
    
    # Final local polish on the best result
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 5:
        f_polished, score_polished = local_polish(
            best_f, start_time, TIME_LIMIT - 2
        )
        if score_polished < best_score:
            best_score = score_polished
            best_f = f_polished
    
    # If SA didn't produce anything good, fall back to constant
    if best_f is None or not np.isfinite(best_score):
        best_f = np.ones(N)
        best_score = compute_score(best_f)
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
