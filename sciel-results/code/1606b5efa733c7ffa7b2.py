"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        seq[i] = (i ^ (i >> 1)).bit_count() % 2
    return seq


def _autoconvolution_peak(signs):
    """Compute the maximum absolute value of the autoconvolution."""
    # Convert to ±1
    s = np.where(signs == 0, 1.0, -1.0)
    # Autoconvolution via FFT
    n = len(s)
    f = np.fft.fft(s, n=2*n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(ac))


def _local_search(signs, max_iter=10000, time_limit=10.0):
    """Greedy local search to minimize autoconvolution peak."""
    best_signs = signs.copy()
    best_val = _autoconvolution_peak(best_signs)
    n = len(best_signs)
    start_time = time.time()
    
    # Greedy: try flipping each position, keep if it improves
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            old = best_signs[i]
            best_signs[i] = 1 - old
            val = _autoconvolution_peak(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[i] = old
    
    # Random restarts with local search
    rng = np.random.RandomState(42)
    while (time.time() - start_time) < time_limit:
        trial = rng.randint(0, 2, size=n).astype(np.int32)
        trial_val = _autoconvolution_peak(trial)
        if trial_val < best_val:
            best_signs = trial.copy()
            best_val = trial_val
            # Local search from this point
            improved2 = True
            while improved2 and (time.time() - start_time) < time_limit:
                improved2 = False
                for i in range(n):
                    old = best_signs[i]
                    best_signs[i] = 1 - old
                    val = _autoconvolution_peak(best_signs)
                    if val < best_val:
                        best_val = val
                        improved2 = True
                    else:
                        best_signs[i] = old
    
    return best_signs, best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    tm = _thue_morse(N)
    
    # Try both polarities
    signs_a = tm.copy()
    signs_b = 1 - tm
    val_a = _autoconvolution_peak(signs_a)
    val_b = _autoconvolution_peak(signs_b)
    
    if val_a <= val_b:
        best_signs = signs_a
        best_val = val_a
    else:
        best_signs = signs_b
        best_val = val_b
    
    # Local search to improve
    if time.time() < deadline - 20.0:
        best_signs, best_val = _local_search(best_signs, time_limit=110.0)
    
    # Final check: ensure we're within deadline
    return best_signs
# EVOLVE-BLOCK-END
