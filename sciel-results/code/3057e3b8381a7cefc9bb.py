"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_peak_autocorr(x):
    """Compute peak absolute autocorrelation including zero lag."""
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(acf))


def _compute_peak_autocorr_excl0(x):
    """Compute peak absolute autocorrelation excluding zero lag."""
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0


def _local_search(x, metric_func, max_iters=100, deadline=None):
    """Greedy local search: flip individual entries to reduce metric."""
    best_x = x.copy()
    best_val = metric_func(best_x)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            x[j] = -x[j]
            val = metric_func(x)
            if val < best_val:
                best_x = x.copy()
                best_val = val
                improved = True
            else:
                x[j] = -x[j]
        if not improved:
            break
    return best_x, best_val


def construct():
    deadline = time.time() + 110
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = np.inf
    
    metrics = [_compute_peak_autocorr, _compute_peak_autocorr_excl0]
    
    for metric in metrics:
        if time.time() > deadline:
            break
        
        for k in [1, 2, 3, 4, 5, 8]:
            if time.time() > deadline:
                break
            
            for trial in range(2):
                if time.time() > deadline:
                    break
                
                x = np.zeros(N)
                indices = rng.choice(N, size=k, replace=False)
                x[indices] = rng.choice([-1, 1], size=k)
                
                x, val = _local_search(x, metric, max_iters=100, deadline=deadline)
                
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
