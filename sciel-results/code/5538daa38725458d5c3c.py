"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(x):
    """Compute peak absolute autocorrelation sidelobe for a +/-1 sequence."""
    # x is array of 1s and -1s
    # autocorrelation at lag k: sum_{i=0}^{N-1-k} x[i]*x[i+k]
    # Use FFT for speed
    n = len(x)
    # Zero-pad to next power of 2 for FFT
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.fft(x, size)
    acf = np.fft.ifft(X * np.conj(X))[:n].real
    # acf[0] is the DC component (= n), acf[1:] are sidelobes
    # Also need negative lags, but for real signals acf is symmetric
    peak = np.max(np.abs(acf[1:]))
    return peak

def _score(x):
    """The upper bound value (lower is better)."""
    peak = _autocorr_peak(x)
    # The value is the peak sidelobe, normalized by N
    # For seed (all ones), peak = N-1 = 399, value = 399/400 = 0.9975
    # Actually, let's just use the raw peak sidelobe
    return peak

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with random initialization
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    best_x = x.copy()
    best_val = _score(best_x)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Random flip
        idx = rng.integers(0, N)
        x[idx] *= -1
        
        new_val = _score(x)
        delta = new_val - best_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_x = x.copy()
            best_val = new_val
        
        T *= cooling
    
    # Final polish: try all single flips
    if time.time() < deadline - 5:
        for i in range(N):
            x_orig = best_x[i]
            best_x[i] *= -1
            val = _score(best_x)
            if val < best_val:
                best_val = val
            else:
                best_x[i] = x_orig
    
    return best_x
# EVOLVE-BLOCK-END
