"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _quasi_periodic_signs(n, alpha=np.sqrt(2), phase=0.0):
    """
    Generate quasi-periodic signed step function using an algebraic frequency.
    s[i] = sign(sin(pi * alpha * i + phase))
    This creates a sequence with dense spectral distribution, avoiding discrete peaks.
    """
    i = np.arange(n, dtype=np.float64)
    # Use sin to create ±1 values
    vals = np.sin(np.pi * alpha * i + phase)
    # Map to 0/1 (where 0 means +1, 1 means -1)
    signs = (vals < 0).astype(np.int32)
    return signs


def _autoconvolution_peak(signs):
    """Compute the maximum absolute value of the autoconvolution."""
    s = np.where(signs == 0, 1.0, -1.0)
    n = len(s)
    f = np.fft.fft(s, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(ac))


def _local_search(signs, time_limit=10.0):
    """
    Greedy local search to minimize autoconvolution peak.
    Uses a single pass per iteration to stay fast.
    """
    best_signs = signs.copy()
    best_val = _autoconvolution_peak(best_signs)
    n = len(best_signs)
    start_time = time.time()
    
    # Multiple passes of greedy flipping
    while (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            old = best_signs[i]
            best_signs[i] = 1 - old
            val = _autoconvolution_peak(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[i] = old
        if not improved:
            break
    
    return best_signs, best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_signs = None
    best_val = np.inf

    # Try several quasi-periodic sequences with different algebraic frequencies and phases
    candidates = []
    
    # Algebraic numbers: sqrt(2), sqrt(3), golden ratio, sqrt(5)
    alphas = [np.sqrt(2), np.sqrt(3), (1 + np.sqrt(5)) / 2, np.sqrt(5), np.sqrt(2) + np.sqrt(3)]
    phases = [0.0, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3]
    
    for alpha in alphas:
        for phase in phases:
            signs = _quasi_periodic_signs(N, alpha, phase)
            val = _autoconvolution_peak(signs)
            candidates.append((val, signs.copy()))
    
    # Sort by value and take the best few
    candidates.sort(key=lambda x: x[0])
    
    # Take top 3 candidates for local search
    for val, signs in candidates[:3]:
        if val < best_val:
            best_val = val
            best_signs = signs.copy()
    
    # Local search on the best candidate
    if best_signs is not None and time.time() < deadline - 20.0:
        remaining_time = min(deadline - 20.0 - time.time(), 110.0)
        best_signs, best_val = _local_search(best_signs, time_limit=remaining_time)
    
    # If we still have time, try local search on the second-best candidate
    if len(candidates) > 1 and time.time() < deadline - 15.0:
        val2, signs2 = candidates[1]
        if val2 < best_val:
            remaining_time = min(deadline - 15.0 - time.time(), 30.0)
            if remaining_time > 5.0:
                signs2, val2 = _local_search(signs2, time_limit=remaining_time)
                if val2 < best_val:
                    best_val = val2
                    best_signs = signs2
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.zeros(N, dtype=np.int32)
    
    return best_signs
# EVOLVE-BLOCK-END
