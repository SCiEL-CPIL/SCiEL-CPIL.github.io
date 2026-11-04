"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p) for prime p."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return 1 if result == 1 else -1

def _max_autocorr_fft(x):
    """Compute max |autocorrelation| for lags 1..N-1 using FFT."""
    N = len(x)
    # Use FFT to compute autocorrelation
    size = 1
    while size < 2 * N:
        size *= 2
    F = np.fft.fft(x, size)
    power = F * np.conj(F)
    autocorr = np.fft.ifft(power).real[:N]
    # autocorr[0] is the energy, we want max of |autocorr[1:]|
    return np.max(np.abs(autocorr[1:]))

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill
    
    # Use Legendre sequence modulo p=401 (prime, p-1=400=N)
    p = 401
    x = np.zeros(N, dtype=np.float64)
    for i in range(1, N + 1):
        x[i-1] = float(_legendre_symbol(i, p))
    
    best_x = x.copy()
    best_score = _max_autocorr_fft(best_x)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_x.astype(np.int8).tolist()
    
    # Try all cyclic shifts
    for shift in range(N):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        shifted = np.roll(x, shift)
        score = _max_autocorr_fft(shifted)
        if score < best_score:
            best_score = score
            best_x = shifted.copy()
    
    # Local search: flip individual bits
    for _round in range(20):
        improved = False
        for i in range(N):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                return best_x.astype(np.int8).tolist()
            trial = best_x.copy()
            trial[i] = -trial[i]
            score = _max_autocorr_fft(trial)
            if score < best_score:
                best_score = score
                best_x = trial
                improved = True
        if not improved:
            break
    
    return best_x.astype(np.int8).tolist()
# EVOLVE-BLOCK-END
