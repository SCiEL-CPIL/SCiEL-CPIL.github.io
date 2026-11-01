"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(h):
    """
    Compute the C3 autoconvolution upper bound.
    For a ±1 sequence, this is based on the third moment of the autocorrelation.
    """
    # Compute linear autocorrelation using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    
    # acf[0] = N (energy), acf[1:] are sidelobes
    # C3 bound: (1/N) * sum_{k=1}^{N-1} |acf[k]|^3 / N^2
    # Or possibly just the sum of cubes normalized
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3


def _compute_max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))


def _local_search_c3(h, deadline):
    """Greedy local search: flip bits to reduce C3 bound."""
    best_h = h.copy()
    best_val = _compute_c3_bound(best_h)
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            h[i] = -h[i]
            val = _compute_c3_bound(h)
            h[i] = -h[i]  # revert
            
            if val < best_val:
                h[i] = -h[i]  # keep flip
                best_val = val
                best_h = h.copy()
                improved = True
        
        if not improved:
            break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with all ones (the seed)
    best_h = np.ones(N)
    best_val = _compute_c3_bound(best_h)
    
    # Try random ±1 sequences
    rng = np.random.default_rng(42)
    for restart in range(10):
        if time.time() > deadline:
            break
        
        h = rng.choice([-1.0, 1.0], size=N)
        val = _compute_c3_bound(h)
        if val < best_val:
            best_h = h.copy()
            best_val = val
    
    # Local search from best found
    if time.time() < deadline - 30:
        result_h, result_val = _local_search_c3(best_h, deadline)
        if result_val < best_val:
            best_h = result_h
            best_val = result_val
    
    return best_h
# EVOLVE-BLOCK-END
