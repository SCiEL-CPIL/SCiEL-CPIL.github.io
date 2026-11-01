"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_sidelobe_max(x):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    n = len(x)
    # Compute autocorrelation via FFT
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # acf[0] is the peak (lag 0), acf[1:] are positive lags
    # For real sequences, autocorrelation is symmetric, so max sidelobe = max(|acf[1:]|)
    return np.max(np.abs(acf[1:]))


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with random ±1 sequence
    np.random.seed(42)
    best_x = np.random.choice([-1.0, 1.0], size=N)
    best_score = _autocorr_sidelobe_max(best_x)
    
    # Simulated annealing
    current_x = best_x.copy()
    current_score = best_score
    
    # Parameters
    T_start = 5.0
    T_end = 0.01
    alpha = 0.9995  # Cooling rate
    T = T_start
    
    while time.time() < deadline:
        # Pick a random position to flip
        idx = np.random.randint(N)
        old_val = current_x[idx]
        current_x[idx] = -old_val
        
        new_score = _autocorr_sidelobe_max(current_x)
        delta = new_score - current_score
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            current_x[idx] = old_val  # Revert
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    # Final polish: try single flips until no improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            old_val = best_x[idx]
            best_x[idx] = -old_val
            new_score = _autocorr_sidelobe_max(best_x)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_x[idx] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
