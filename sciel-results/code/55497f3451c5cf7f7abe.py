import numpy as np
import time

N = 400
DEADLINE = 120.0
# Use a fine grid for accurate autoconvolution calculation
# M = 4 * N provides 4 points per step interval, which is sufficient for 
# resolving the peak of the autoconvolution for optimization purposes.
M = 4 * N 
dx = 0.5 / M

def _compute_c3_fast(signed_step):
    """
    Compute the max absolute value of the autoconvolution of the signed step function.
    Uses FFT on a fine grid.
    """
    # Expand step function to fine grid
    # signed_step is shape (N,), values are +/- 1
    f_fine = np.repeat(signed_step, M // N).astype(np.float64)
    
    # Autoconvolution via FFT
    # Zero-pad to 2*M to avoid circular convolution effects
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    # Initialize with random signs. 
    # All-ones gives C3=0.5. Random gives much lower C3 typically.
    np.random.seed(42)
    current_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    current_c3 = _compute_c3_fast(current_step)
    
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Simulated Annealing parameters
    # Initial temperature should be comparable to typical energy differences
    T0 = 0.01 
    T = T0
    # Cooling factor
    alpha = 0.9995
    
    # We will run until time limit
    # Estimate iterations: ~1000-2000 per second? 
    # FFT size 3200 is very fast. Let's just loop until deadline.
    
    last_improve_time = start_time
    no_improve_count = 0
    patience = 5000 # Number of steps without improvement before breaking
    
    while True:
        # Check deadline with safety margin
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate move: single flip
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3_fast(current_step)
        delta = new_c3 - current_c3
        
        # Metropolis acceptance
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_step[idx] = old_val
            no_improve_count += 1
        
        # Cool down
        T *= alpha
        
        # If stuck for too long, reheat or break
        if no_improve_count > patience:
            # Reheat to escape local minimum
            T = T0 * 0.5
            no_improve_count = 0
            
        # Optional: if T is too low, stop
        if T < 1e-6:
            # Do a greedy polish
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    old_val = current_step[i]
                    current_step[i] = -old_val
                    c3_try = _compute_c3_fast(current_step)
                    if c3_try < current_c3:
                        current_c3 = c3_try
                        if c3_try < best_c3:
                            best_c3 = c3_try
                            best_step = current_step.copy()
                        improved = True
                    else:
                        current_step[i] = old_val
                if not improved:
                    break
            break

    return best_step
