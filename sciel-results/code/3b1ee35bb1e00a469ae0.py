import numpy as np
import time

N = 400
DEADLINE = 118.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the step function.
    The step function is defined on [-0.25, 0.25] with N equal intervals.
    We use a fine grid to approximate the continuous autoconvolution via FFT.
    """
    # Use a grid resolution of 4 points per interval for accuracy
    M = 4 * N
    # Length of the interval is 0.5. So dx = 0.5 / M
    dx = 0.5 / M
    
    # Expand the step function to the fine grid
    # Each step height is repeated M//N = 4 times
    f_fine = np.repeat(signed_step, 4).astype(np.float64)
    
    # Perform FFT on a zero-padded array to compute linear convolution (autoconvolution)
    # The length of the autoconvolution is 2*M - 1. We pad to 2*M for efficient FFT.
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    
    # The autoconvolution calculated by FFT is a sum, so we multiply by dx to get the integral
    # Note: The standard convolution sum approximates the integral with factor dx.
    autoconv *= dx
    
    # The support of the autoconvolution is effectively [-0.5, 0.5] shifted.
    # We take the max absolute value over the entire computed array.
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    # Initialize with a random sign pattern (zero mean is generally better for minimizing L_infty of autoconv)
    # But we can start with a specific pattern or random. Random is safer to avoid bias.
    best_step = np.random.choice([-1, 1], size=N)
    best_c3 = _compute_c3(best_step)
    
    # Local search with Simulated Annealing
    np.random.seed(42)
    
    max_iterations = 100000
    no_improve_count = 0
    patience = 20000
    
    # Temperature for occasional uphill moves
    T = 1e-3
    T_min = 1e-5
    T_decay = 0.99995
    
    # Track the best ever found
    global_best_step = best_step.copy()
    global_best_c3 = best_c3
    
    for iteration in range(max_iterations):
        # Check deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Move: Flip 1 or 2 random elements
        # Multi-flip moves help escape local minima
        num_flips = 1 if np.random.random() > 0.2 else 2
        indices = np.random.choice(N, size=num_flips, replace=False)
        
        old_values = best_step[indices].copy()
        best_step[indices] = -old_values
        
        new_c3 = _compute_c3(best_step)
        
        delta = new_c3 - best_c3
        
        # Acceptance criteria
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            # Accept move
            best_c3 = new_c3
            no_improve_count = 0
            
            # Update global best
            if best_c3 < global_best_c3:
                global_best_c3 = best_c3
                global_best_step = best_step.copy()
        else:
            # Reject move
            best_step[indices] = old_values
            no_improve_count += 1
        
        # Cool down
        T = max(T_min, T * T_decay)
        
        if no_improve_count >= patience:
            break
    
    # Final polish: Greedy pass to ensure no single flip improves
    if time.time() - start_time < DEADLINE - 5.0:
        current_step = global_best_step.copy()
        current_c3 = global_best_c3
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2.0:
            improved = False
            for idx in range(N):
                if time.time() - start_time > DEADLINE - 2.0:
                    break
                old_val = current_step[idx]
                current_step[idx] = -old_val
                new_c3 = _compute_c3(current_step)
                if new_c3 < current_c3:
                    current_c3 = new_c3
                    global_best_c3 = current_c3
                    global_best_step = current_step.copy()
                    improved = True
                else:
                    current_step[idx] = old_val
    
    return global_best_step
