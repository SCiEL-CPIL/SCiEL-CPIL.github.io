# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Full autocorrelation via FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:n]))


def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf_full = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf_full[1:n]


def _update_acf(acf, h, i):
    """
    Compute the new ACF after flipping h[i].
    Returns the new ACF array.
    """
    n = len(h)
    # Delta for each lag k (1 to n-1)
    # acf index k-1 corresponds to lag k
    # Delta[k-1] = -2 * h[i] * (h[i+k] + h[i-k])
    
    # Create a copy of the current ACF
    new_acf = acf.copy()
    
    hi = h[i]
    
    # We need to iterate k from 1 to n-1
    # Vectorized approach:
    # k = np.arange(1, n)
    # idx1 = i + k
    # idx2 = i - k
    # mask1 = (idx1 >= 0) & (idx1 < n)
    # mask2 = (idx2 >= 0) & (idx2 < n)
    # vals1 = np.where(mask1, h[idx1], 0.0)
    # vals2 = np.where(mask2, h[idx2], 0.0)
    # deltas = -2.0 * hi * (vals1 + vals2)
    # new_acf += deltas
    
    k = np.arange(1, n)
    idx1 = i + k
    idx2 = i - k
    
    mask1 = (idx1 >= 0) & (idx1 < n)
    mask2 = (idx2 >= 0) & (idx2 < n)
    
    # Use zeros for out of bounds
    vals1 = np.zeros(n-1)
    vals2 = np.zeros(n-1)
    
    if np.any(mask1):
        vals1[mask1] = h[idx1[mask1]]
    if np.any(mask2):
        vals2[mask2] = h[idx2[mask2]]
        
    deltas = -2.0 * hi * (vals1 + vals2)
    new_acf += deltas
    
    return new_acf


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    Uses Simulated Annealing with incremental ACF updates.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin
    
    n = N
    # Initial random sequence
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=n)
    
    # Initial ACF
    acf = _compute_acf(h)
    current_peak = np.max(np.abs(acf))
    
    best_h = h.copy()
    best_peak = current_peak
    
    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.1
    # Number of steps per temperature? Or just time-based?
    # Let's do time-based with a fixed step count per iteration to control speed.
    
    # Estimate steps per second
    # One step: update ACF (O(N)), check peak (O(N)), maybe accept.
    # N=400. O(800) ops. Very fast.
    # We can do ~100,000 steps per second easily in Python if vectorized.
    # Let's try a fixed number of iterations per "epoch" and cool down.
    
    # Better: Just run until time limit, cooling T over time.
    # Total time ~ 135s.
    # Let's aim for ~500,000 steps.
    
    # Adaptive cooling: T = T_start * (T_end/T_start)**(elapsed / total_time)
    total_time = deadline - start_time
    if total_time <= 0:
        return best_h
        
    steps_done = 0
    # Pre-allocate index array for random selection? 
    # rng.integers(0, n) is fast enough.
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        # Cool down
        progress = elapsed / total_time
        if progress > 1.0:
            progress = 1.0
        T = T_start * (T_end / T_start) ** progress
        
        # Batch of steps to reduce Python loop overhead?
        # Let's do 100 steps per loop iteration.
        for _ in range(100):
            if time.time() > deadline:
                break
                
            i = rng.integers(0, n)
            new_acf = _update_acf(acf, h, i)
            new_peak = np.max(np.abs(new_acf))
            
            delta_peak = new_peak - current_peak
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_peak < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta_peak / T)
                else:
                    prob = 0.0
                accept = rng.random() < prob
                
            if accept:
                # Flip h[i]
                h[i] = -h[i]
                acf = new_acf
                current_peak = new_peak
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
            else:
                # Rejected, keep old state
                pass
                
            steps_done += 1
            
    return best_h
# EVOLVE-BLOCK-END
