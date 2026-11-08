import numpy as np
import time
import random

N = 400
DEADLINE = 140.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # signed_step is an array of N values in {-1, 1}
    h = 1.0 / N
    
    # Use FFT to compute the autoconvolution efficiently
    # We need to pad to at least 2*N to avoid circular convolution effects
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    # Build the signal on a fine grid
    # Each interval has width h = 1/N
    # We'll represent the function on a grid of 'size' points
    f = np.zeros(size)
    for i in range(N):
        start = i * (size // N)
        end = (i + 1) * (size // N)
        f[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale by the width of each grid cell
    # The integral is approximated by sum * dx, where dx = 0.5 / size
    dx = 0.5 / size
    autoconv *= dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    max_iterations = 200000
    no_improve_count = 0
    patience = 2000
    
    for iteration in range(max_iterations):
        # Check deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            best_step[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count >= patience:
            break
    
    return best_step
