import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    Optimized using rfft and vectorized array construction.
    """
    M = 4 * N
    dx = 0.5 / M
    
    # Vectorized construction of the fine-grained function
    # signed_step has length N. Each element repeats M/N times.
    # M/N = 4.
    f_fine = np.repeat(signed_step, M // N)
    
    # Use rfft for real input
    fft_f = np.fft.rfft(f_fine, n=2*M)
    
    # Power spectrum
    power = fft_f * np.conj(fft_f)
    
    # Inverse FFT
    autoconv = np.fft.irfft(power, n=2*M) * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    current_step = np.ones(N, dtype=np.int8)
    current_c3 = _compute_c3(current_step)
    
    # Track best solution
    best_step = current_step.copy()
    best_c3 = current_c3
    
    np.random.seed(42)
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 0.001
    total_iters = 20000
    
    # Geometric cooling factor
    cooling_factor = (T_end / T_start) ** (1.0 / total_iters)
    T = T_start
    
    for t in range(total_iters):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        delta_e = new_c3 - current_c3
        
        # Acceptance criterion
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            # Accept move
            current_c3 = new_c3
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Reject move: revert
            current_step[idx] = old_val
            
        # Cool down
        T *= cooling_factor
        
    return best_step
