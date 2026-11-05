"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to +/-1."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # popcount parity
        seq[i] = bin(i).count('1') % 2
    return 2 * seq.astype(np.float64) - 1


def _compute_c3_bound(x):
    """Compute the peak absolute value of the linear autoconvolution."""
    n = len(x)
    conv = np.correlate(x, x, mode='full')
    return np.max(np.abs(conv))


def _compute_c3_bound_fast(x):
    """Compute the peak absolute value using FFT for speed."""
    n = len(x)
    # Use next power of 2 for FFT efficiency
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.fft(x, n=size)
    conv = np.fft.ifft(X * X).real[:2*n-1]
    return np.max(np.abs(conv))


def _logistic_sequence(r, x0, n):
    """Generate a sequence from the logistic map, mapped to +/-1."""
    x = np.zeros(n, dtype=np.float64)
    val = x0
    for i in range(n):
        x[i] = 1.0 if val > 0.5 else -1.0
        val = r * val * (1.0 - val)
    return x


def _local_search(x, time_limit=10):
    """Local search: flip individual bits to reduce autoconvolution peak."""
    x = x.copy()
    best_peak = _compute_c3_bound_fast(x)
    start_time = time.time()
    n = len(x)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Randomize the order of flips for diversity
        order = np.random.permutation(n)
        for i in order:
            if (time.time() - start_time) > time_limit:
                break
            x[i] *= -1
            new_peak = _compute_c3_bound_fast(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1  # revert
    
    return x, best_peak


def _pairwise_flip_search(x, time_limit=10):
    """Pairwise flip search: flip pairs of bits to reduce autoconvolution peak."""
    x = x.copy()
    best_peak = _compute_c3_bound_fast(x)
    start_time = time.time()
    n = len(x)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try random pairs
        for _ in range(500):
            if (time.time() - start_time) > time_limit:
                break
            i, j = np.random.choice(n, size=2, replace=False)
            x[i] *= -1
            x[j] *= -1
            new_peak = _compute_c3_bound_fast(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1
                x[j] *= -1  # revert
    
    return x, best_peak


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_x = None
    best_peak = np.inf
    
    # Strategy 1: Thue-Morse based initialization
    x_tm = _thue_morse(N)
    x_refined, peak = _local_search(x_tm, time_limit=15)
    if peak < best_peak:
        best_peak = peak
        best_x = x_refined
    
    # Strategy 2: Logistic map based initialization with various parameters
    if (time.time() - start_time) < deadline - 30:
        configs = [
            (4.0, 0.5),
            (3.999, 0.3),
            (4.0, 0.7),
            (3.99, 0.1),
            (4.0, 0.9),
            (3.995, 0.5),
            (3.998, 0.2),
            (3.997, 0.8),
        ]
        
        for r, x0 in configs:
            if (time.time() - start_time) > deadline - 20:
                break
            
            x = _logistic_sequence(r, x0, N)
            time_remaining = deadline - (time.time() - start_time) - 10
            if time_remaining > 5:
                x_refined, peak = _local_search(x, time_limit=min(8, time_remaining // 2))
            else:
                x_refined, peak = x, _compute_c3_bound_fast(x)
            
            if peak < best_peak:
                best_peak = peak
                best_x = x_refined
    
    # Strategy 3: Random initialization with local search
    if (time.time() - start_time) < deadline - 15:
        for _ in range(3):
            if (time.time() - start_time) > deadline - 10:
                break
            x_rand = np.random.choice([-1, 1], size=N)
            time_remaining = deadline - (time.time() - start_time) - 5
            if time_remaining > 5:
                x_refined, peak = _local_search(x_rand, time_limit=min(8, time_remaining // 2))
            else:
                x_refined, peak = x_rand, _compute_c3_bound_fast(x_rand)
            
            if peak < best_peak:
                best_peak = peak
                best_x = x_refined
    
    # Final polish: pairwise flip search on the best solution
    if best_x is not None and (time.time() - start_time) < deadline - 5:
        time_for_pairwise = min(10, deadline - (time.time() - start_time) - 3)
        if time_for_pairwise > 3:
            x_final, peak_final = _pairwise_flip_search(best_x, time_limit=time_for_pairwise)
            if peak_final < best_peak:
                best_peak = peak_final
                best_x = x_final
    
    if best_x is None:
        # Fallback
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
