import numpy as np
import time

N = 400
DEADLINE = 120.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5


def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    Vectorized: builds fine grid via repeat, computes autoconvolution via FFT.
    """
    # Build fine grid: each interval i has M/N = 4 samples
    f_fine = np.repeat(signed_step, M // N).astype(np.float64)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # C3 = max |autoconv|
    return np.max(np.abs(autoconv))


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Simulated annealing parameters
    T_start = 0.05  # initial temperature (scale of C3 changes)
    T_end = 0.001
    cooling_rate = 0.9995  # per iteration
    
    # Current state for SA
    current_step = best_step.copy()
    current_c3 = best_c3
    
    T = T_start
    no_improve_count = 0
    patience = 8000
    
    while time.time() - start_time < DEADLINE - 3.0:
        # Multi-flip move: flip 1-4 random positions
        n_flips = rng.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.10])
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        
        # Apply flips
        old_vals = current_step[flip_indices].copy()
        current_step[flip_indices] = -current_step[flip_indices]
        
        new_c3 = _compute_c3(current_step)
        delta = new_c3 - current_c3
        
        # Acceptance criterion
        if delta < 0 or (T > 1e-8 and rng.random() < np.exp(-delta / T)):
            # Accept move
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject move
            current_step[flip_indices] = old_vals
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        if no_improve_count >= patience:
            break
    
    return best_step
