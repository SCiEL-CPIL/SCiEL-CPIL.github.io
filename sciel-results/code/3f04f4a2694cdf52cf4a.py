import numpy as np
import time

N = 400
DEADLINE = 118.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max_t |autoconv(t)|
    using FFT on a fine grid.
    """
    M = 4 * N
    dx = 0.5 / M
    f_fine = np.repeat(signed_step, M // N).astype(np.float64)
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M) * dx
    return np.max(np.abs(autoconv))

def _compute_c3_fast(signed_step):
    """
    Faster C3 computation using precomputed FFT structure.
    """
    M = 4 * N
    dx = 0.5 / M
    f_fine = np.repeat(signed_step, M // N).astype(np.float64)
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M) * dx
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Local search with simulated annealing-like acceptance
    np.random.seed(42)
    
    max_iterations = 50000
    no_improve_count = 0
    patience = 8000
    
    # Temperature for occasional uphill moves
    T = 1e-4
    T_min = 1e-6
    
    for iteration in range(max_iterations):
        # Check deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3_fast(best_step)
        
        delta = new_c3 - best_c3
        if delta < 0:
            # Improvement
            best_c3 = new_c3
            no_improve_count = 0
        else:
            # Accept with probability exp(-delta/T)
            if np.random.random() < np.exp(-delta / T):
                best_c3 = new_c3
                no_improve_count = 0
            else:
                best_step[idx] = old_val
                no_improve_count += 1
        
        # Cool down
        T = max(T_min, T * 0.9999)
        
        if no_improve_count >= patience:
            break
    
    # Final polish: greedy pass to ensure no single flip improves
    if time.time() - start_time < DEADLINE - 5.0:
        for idx in range(N):
            if time.time() - start_time > DEADLINE - 2.0:
                break
            old_val = best_step[idx]
            best_step[idx] = -old_val
            new_c3 = _compute_c3_fast(best_step)
            if new_c3 < best_c3:
                best_c3 = new_c3
            else:
                best_step[idx] = old_val
    
    return best_step
