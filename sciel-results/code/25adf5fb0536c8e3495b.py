"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(x):
    """Compute the maximum absolute value of the autocorrelation of x."""
    n = len(x)
    # Full autocorrelation via FFT
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # The autocorrelation at lag 0 is n (sum of squares = n for ±1)
    # We want the max |acf[k]| for k != 0, normalized
    # Upper bound on C3 relates to max |acf[k]| / n for k != 0
    peak = np.max(np.abs(acf[1:]))
    return peak / n


def _greedy_local_search(x, deadline):
    """Greedy local search: flip bits that reduce the autocorrelation peak."""
    n = len(x)
    current_peak = _autocorr_peak(x)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_idx = -1
        for i in range(n):
            if time.time() >= deadline:
                break
            # Try flipping bit i
            x[i] = -x[i]
            new_peak = _autocorr_peak(x)
            gain = current_peak - new_peak
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            x[i] = -x[i]  # revert
        if best_idx >= 0:
            x[best_idx] = -x[best_idx]
            current_peak -= best_gain
            improved = True
    return x, current_peak


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_x = np.ones(N)
    best_score = _autocorr_peak(best_x)
    
    # Multiple restarts with diverse initializations
    rng = np.random.default_rng(42)
    
    # Restart 1: all ones
    x = np.ones(N)
    x, score = _greedy_local_search(x, deadline)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    # Restart 2: random
    if time.time() < deadline:
        x = rng.choice([-1.0, 1.0], size=N)
        x, score = _greedy_local_search(x, deadline)
        if score < best_score:
            best_x = x.copy()
            best_score = score
    
    # Restart 3: Thue-Morse-like pattern
    if time.time() < deadline:
        x = np.ones(N)
        for i in range(N):
            # Count bits in i
            x[i] = 1.0 if bin(i).count('1') % 2 == 0 else -1.0
        x, score = _greedy_local_search(x, deadline)
        if score < best_score:
            best_x = x.copy()
            best_score = score
    
    # Restart 4: Legendre-like (quadratic phase)
    if time.time() < deadline:
        x = np.ones(N)
        for i in range(N):
            x[i] = 1.0 if (i * i) % 2 == 0 else -1.0
        x, score = _greedy_local_search(x, deadline)
        if score < best_score:
            best_x = x.copy()
            best_score = score
    
    # Restart 5: another random
    if time.time() < deadline:
        x = rng.choice([-1.0, 1.0], size=N)
        x, score = _greedy_local_search(x, deadline)
        if score < best_score:
            best_x = x.copy()
            best_score = score
    
    # Final polish: one more round of greedy if time permits
    if time.time() < deadline - 2:
        x, score = _greedy_local_search(best_x, deadline)
        if score < best_score:
            best_x = x.copy()
            best_score = score
    
    return best_x
# EVOLVE-BLOCK-END
