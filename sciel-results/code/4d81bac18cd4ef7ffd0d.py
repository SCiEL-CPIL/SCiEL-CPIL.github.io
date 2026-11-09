"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count number of 1s in binary representation
        result[i] = bin(i).count('1') % 2
    return result


def _autocorr_sidelobe_max(x):
    """Compute maximum absolute value of autocorrelation sidelobes."""
    # Compute autocorrelation using FFT
    n = len(x)
    fft_size = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, fft_size)
    acf = np.fft.irfft(X * np.conj(X), fft_size)[:n]
    # Sidelobes are all lags except lag 0
    return np.max(np.abs(acf[1:]))


def _score(x):
    """Score function: upper bound on C3 based on autocorrelation sidelobes."""
    return _autocorr_sidelobe_max(x)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with Thue-Morse sequence
    x = _thue_morse(N).astype(np.float64)
    # Convert to ±1
    x = 1 - 2 * x  # 0->1, 1->-1
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Simulated Annealing
    T = 0.5  # Initial temperature
    cooling = 0.9995
    T_min = 0.001
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Random single-bit flip
        idx = np.random.randint(N)
        x_new = x.copy()
        x_new[idx] *= -1
        
        score_new = _score(x_new)
        delta = score_new - best_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x = x_new
            if score_new < best_score:
                best_score = score_new
                best_x = x_new.copy()
        
        T *= cooling
    
    # Final polish: try all single-bit flips from best
    if time.time() < deadline - 5:
        for idx in range(N):
            if time.time() > deadline:
                break
            x_test = best_x.copy()
            x_test[idx] *= -1
            s = _score(x_test)
            if s < best_score:
                best_score = s
                best_x = x_test
    
    return best_x
# EVOLVE-BLOCK-END
