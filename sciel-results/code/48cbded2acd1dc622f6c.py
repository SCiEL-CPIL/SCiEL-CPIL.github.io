"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_peak_autocorr(h):
    """Compute the maximum absolute value of the linear autocorrelation at non-zero lags."""
    n = len(h)
    # Use FFT for linear autocorrelation
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    auto = np.fft.irfft(power, n=fft_size)
    # auto[0] = sum(h^2), auto[1..n-1] = R[1..n-1]
    # For real sequences, R[-k] = R[k], so we only need positive lags
    if n <= 1:
        return 0.0
    peak = np.max(np.abs(auto[1:n]))
    return peak


def _local_search(h, time_limit, allow_zero=True):
    """Local search: flip signs or toggle zero/non-zero to reduce peak autocorrelation."""
    n = len(h)
    h = h.copy()
    current_peak = _compute_peak_autocorr(h)
    best_peak = current_peak
    best_h = h.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try modifying each position
        for i in range(n):
            old_val = h[i]
            # Try different values
            candidates = []
            if allow_zero:
                candidates = [-1.0, 0.0, 1.0]
            else:
                candidates = [-1.0, 1.0]
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                h[i] = new_val
                new_peak = _compute_peak_autocorr(h)
                if new_peak < current_peak - 1e-12:
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_h = h.copy()
                    break
                else:
                    h[i] = old_val
            
            if improved:
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_h, best_peak


def _sparse_init(rng, density):
    """Initialize with sparse non-zero values."""
    n = N
    h = np.zeros(n)
    # Select random positions to be non-zero
    num_nonzero = max(1, int(n * density))
    indices = rng.choice(n, size=num_nonzero, replace=False)
    h[indices] = rng.choice([-1.0, 1.0], size=num_nonzero)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_peak = np.inf
    
    # Try different sparsity levels
    densities = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    
    for density in densities:
        if time.time() >= deadline:
            break
        
        # Multiple random restarts for each density
        n_restarts = 5
        time_per_restart = max(0.5, (deadline - start_time - 5.0) / (n_restarts * len(densities)))
        
        for restart in range(n_restarts):
            if time.time() >= deadline:
                break
            
            h_init = _sparse_init(rng, density)
            remaining = deadline - time.time() - 5.0
            if remaining <= 0:
                break
            time_limit = min(remaining, time_per_restart)
            
            h_opt, peak = _local_search(h_init, time_limit, allow_zero=True)
            
            if peak < best_peak:
                best_peak = peak
                best_h = h_opt.copy()
    
    # Final polish with more iterations on the best solution
    if best_h is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 0:
            h_polished, peak_polished = _local_search(best_h, remaining, allow_zero=True)
            if peak_polished < best_peak:
                best_peak = peak_polished
                best_h = h_polished
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
