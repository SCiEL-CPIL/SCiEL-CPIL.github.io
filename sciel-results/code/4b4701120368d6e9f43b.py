"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_max_abs(h):
    """Compute max absolute off-peak autocorrelation for a +/-1 sequence."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n > 1:
        return float(np.max(np.abs(acf[1:n])))
    return 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with random +/-1 sequence
        h = rng.choice([-1.0, 1.0], size=N)
        current_score = _autocorr_max_abs(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Simulated annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        patience = 0
        best_in_run = current_score
        
        while time.time() < deadline - 5.0:
            idx = rng.integers(0, N)
            
            old_val = h[idx]
            h[idx] = -old_val
            
            new_score = _autocorr_max_abs(h)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_in_run:
                    best_in_run = new_score
                    patience = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h.copy()
                else:
                    patience += 1
            else:
                h[idx] = old_val
                patience += 1
            
            T = max(T_min, T * cooling)
            
            # Restart if stuck too long
            if patience > 5000:
                break
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
