import numpy as np
import time
import random
from scipy.optimize import minimize

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
    
    # Define the objective function for continuous optimization
    # The input is a real-valued vector x of length N.
    # We map x to a step function by thresholding: s_i = 1 if x_i >= 0 else -1.
    # However, for the optimizer to work well, we can try to minimize the C3 of the step function
    # derived from x. But C3 is non-differentiable.
    # Alternative: Minimize a continuous proxy.
    # Let's stick to the discrete problem but use a different search strategy.
    # The prompt suggests: "Represent the step function as a binary vector and optimize the C3 bound using a simulated annealing schedule".
    # The current program uses deterministic hill climbing.
    # Let's implement Simulated Annealing.
    
    # We will use a faster C3 calculation for the inner loop if possible, 
    # but for now we reuse _compute_c3. To make SA feasible in 120s, 
    # we need to be efficient.
    
    # Let's try to speed up _compute_c3 by pre-computing the FFT of the "1" vector? No, it changes.
    # We can pre-compute the indices.
    
    size = 1 << (2 * N - 1).bit_length()
    dx = 0.5 / size
    
    # Pre-compute the mapping from index i in [0, N) to slice in [0, size)
    # This is already fast enough in numpy if we vectorize the update?
    # No, we flip one element at a time.
    
    # Let's use a slightly different metric for the SA acceptance to be faster?
    # No, must minimize the actual C3.
    
    # To make SA work, we need to evaluate many candidates.
    # _compute_c3 takes ~0.5ms? Let's assume 1ms. 120s / 1ms = 120,000 iterations.
    # This is feasible.
    
    current_step = np.ones(N, dtype=np.int8)
    current_c3 = best_c3
    
    T_start = 0.1  # Initial temperature
    T_end = 1e-6   # Final temperature
    alpha = 0.99   # Cooling rate
    
    # We need a "delta" estimation or just full re-computation?
    # Full re-computation is safer.
    
    steps = 100000
    for i in range(steps):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        # Calculate current temperature
        # Geometric cooling
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        old_val = current_step[idx]
        new_val = -old_val
        current_step[idx] = new_val
        
        # Compute new C3
        new_c3 = _compute_c3(current_step)
        
        delta = new_c3 - current_c3
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Worse, accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_c3 = new_c3
                else:
                    # Revert
                    current_step[idx] = old_val
    
    return best_step
