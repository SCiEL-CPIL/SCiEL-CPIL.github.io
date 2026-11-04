"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autocorr_peak(f):
    """Compute the maximum absolute autocorrelation excluding lag 0."""
    n = len(f)
    # Use FFT for fast autocorrelation
    f_float = f.astype(np.float64)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    F = np.fft.rfft(f_float, n=fft_size)
    acf = np.fft.irfft(F * np.conj(F), n=fft_size)[:n]
    # acf[0] is the sum of squares = n, exclude it
    return np.max(np.abs(acf[1:]))

def _local_search(f, max_iters=5000, deadline=None):
    """Flip individual bits to reduce peak autocorrelation."""
    n = len(f)
    best = f.copy()
    best_peak = _autocorr_peak(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Try flipping a random position
        i = random.randint(0, n - 1)
        f[i] = -f[i]
        new_peak = _autocorr_peak(f)
        if new_peak < best_peak:
            best = f.copy()
            best_peak = new_peak
        else:
            f[i] = -f[i]  # revert
    return best, best_peak

def _simulated_annealing(f, max_iters=10000, T0=1.0, T1=0.01, deadline=None):
    """Simulated annealing for ±1 sequence optimization."""
    n = len(f)
    current = f.copy()
    current_peak = _autocorr_peak(current)
    best = current.copy()
    best_peak = current_peak
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        T = T0 * (T1 / T0) ** (it / max_iters)
        i = random.randint(0, n - 1)
        current[i] = -current[i]
        new_peak = _autocorr_peak(current)
        delta = new_peak - current_peak
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_peak = new_peak
            if new_peak < best_peak:
                best = current.copy()
                best_peak = new_peak
        else:
            current[i] = -current[i]  # revert
    return best, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Multiple random starts with local search and simulated annealing
    best_overall = None
    best_overall_peak = np.inf
    
    random.seed(42)
    np.random.seed(42)
    
    # Try several initializations
    num_starts = 20
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if s == 0:
            # All ones baseline
            f = np.ones(N, dtype=np.int8)
        elif s == 1:
            # Random ±1
            f = np.random.choice([-1, 1], size=N).astype(np.int8)
        elif s == 2:
            # Quadratic residue-like pattern
            f = np.zeros(N, dtype=np.int8)
            for i in range(N):
                f[i] = 1 if (i * i) % 4 < 2 else -1
        else:
            # Perturbed best so far
            if best_overall is not None:
                f = best_overall.copy()
                # Flip a few random positions
                for _ in range(random.randint(5, 20)):
                    f[random.randint(0, N - 1)] *= -1
            else:
                f = np.random.choice([-1, 1], size=N).astype(np.int8)
        
        # Phase 1: Local search
        f_ls, peak_ls = _local_search(f, max_iters=3000, deadline=deadline)
        
        # Phase 2: Simulated annealing from local search result
        f_sa, peak_sa = _simulated_annealing(f_ls, max_iters=5000, T0=1.0, T1=0.01, deadline=deadline)
        
        if peak_sa < best_overall_peak:
            best_overall = f_sa
            best_overall_peak = peak_sa
        
        # Also try SA from the local search result directly
        if time.time() < deadline:
            f_sa2, peak_sa2 = _simulated_annealing(f_ls, max_iters=3000, T0=0.5, T1=0.005, deadline=deadline)
            if peak_sa2 < best_overall_peak:
                best_overall = f_sa2
                best_overall_peak = peak_sa2
    
    if best_overall is None:
        return np.ones(N, dtype=np.int8)
    
    return best_overall
# EVOLVE-BLOCK-END
