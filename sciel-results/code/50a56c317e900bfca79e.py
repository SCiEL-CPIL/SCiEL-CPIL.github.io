"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
DELTA = 1.0 / 200.0
N_GRID = 2048

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(f):
    """Compute C3 = 2 * max_t |f*f(t)| using FFT."""
    x = np.linspace(-0.25, 0.25, N_GRID, endpoint=False)
    idx = np.floor((x + 0.25) / DELTA).astype(int)
    idx = np.clip(idx, 0, N-1)
    f_fine = f[idx]
    
    size = 2 * N_GRID
    F = np.fft.rfft(f_fine, size)
    acf = np.fft.irfft(F * np.conj(F), size)
    dx = 0.5 / N_GRID
    acf_scaled = acf * dx
    
    peak = np.max(np.abs(acf_scaled))
    return 2.0 * peak

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    # Initialize with several random restarts
    best_f = None
    best_score = float('inf')
    
    for _ in range(50):
        f = rng.choice([-1, 1], size=N).astype(float)
        s = _compute_c3(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Simulated annealing with periodic local search
    T = 1.0
    T_min = 1e-5
    cooling = 0.99995
    
    current_f = best_f.copy()
    current_score = best_score
    
    iteration = 0
    while time.time() < deadline:
        # Simulated annealing step
        idx = rng.integers(0, N)
        old_val = current_f[idx]
        current_f[idx] = -old_val
        
        new_score = _compute_c3(current_f)
        delta = new_score - current_score
        
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / max(T, 1e-10))):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
        else:
            current_f[idx] = old_val
        
        T *= cooling
        iteration += 1
        
        # Local search every 500 iterations
        if iteration % 500 == 0 and time.time() < deadline - 2:
            for _ in range(10):
                improved = False
                for i in range(N):
                    old_val = current_f[i]
                    current_f[i] = -old_val
                    s = _compute_c3(current_f)
                    if s < current_score - 1e-12:
                        current_score = s
                        if s < best_score:
                            best_score = s
                            best_f = current_f.copy()
                        improved = True
                    else:
                        current_f[i] = old_val
                if not improved:
                    break
    
    return best_f
# EVOLVE-BLOCK-END
