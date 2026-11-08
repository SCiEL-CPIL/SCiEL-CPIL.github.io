import numpy as np
import time
import math

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    """
    M = 4 * N
    dx = (0.5) / M
    
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with random signed steps
    np.random.seed(42)
    current_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    current_c3 = _compute_c3(current_step)
    
    # Track best solution
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Simulated annealing parameters
    T_initial = 0.1  # Initial temperature (tuned based on typical C3 values)
    T_final = 1e-6   # Final temperature
    cooling_rate = 0.999
    
    T = T_initial
    
    # Run simulated annealing until deadline
    while time.time() - start_time < DEADLINE - 5.0:
        # Choose a random index to flip
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        # Compute new C3
        new_c3 = _compute_c3(current_step)
        
        # Accept or reject based on simulated annealing criterion
        delta = new_c3 - current_c3
        if delta < 0:
            # Always accept improvements
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Accept uphill moves with probability exp(-delta/T)
            if T > 0 and np.random.random() < math.exp(-delta / T):
                current_c3 = new_c3
            else:
                # Revert the move
                current_step[idx] = old_val
        
        # Cool down
        T *= cooling_rate
    
    return best_step
