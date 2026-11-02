import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_c3_bound(x):
    """
    Compute the upper bound on C3.
    Based on the task description, this is likely the maximum absolute value
    of the periodic autocorrelation away from lag 0.
    C(k) = sum_{i=0}^{N-1} x[i] * x[(i+k) % N]
    We return max_{k != 0} |C(k)|.
    """
    x = np.asarray(x)
    # Compute periodic autocorrelation using FFT
    # x is real, so we can use rfft/irfft or just fft/real
    # Autocorrelation via FFT: IFFT(|FFT(x)|^2)
    # Note: np.fft.ifft(np.abs(np.fft.fft(x))**2) gives circular autocorrelation.
    # The result is real, but might have tiny imaginary parts.
    f = np.fft.fft(x)
    acf = np.real(np.fft.ifft(np.abs(f)**2))
    
    # acf[0] is the energy N.
    # We want max absolute value for k != 0.
    # Due to symmetry, acf[k] == acf[N-k].
    # We can just take max of absolute values of acf[1:]
    # But to be safe and fast, just take max of abs(acf[1:])
    max_abs = np.max(np.abs(acf[1:]))
    return max_abs

def construct():
    """
    Search for a signed step function that minimizes the C3 upper bound.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialize with a random sequence
    # Sometimes starting with a specific pattern helps, but random is fine with good SA.
    best_x = rng.choice([-1, 1], size=N)
    best_score = _compute_c3_bound(best_x)
    
    current_x = best_x.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    # We want to minimize the max autocorrelation.
    # Start with a high temperature to explore, cool down to exploit.
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Pre-allocate a temporary array for FFT to avoid overhead? 
    # np.fft.fft allocates internally. We can try to reuse buffers if we go low-level, 
    # but for N=400, it's small enough.
    
    while time.time() < deadline:
        # Pick a random position to flip
        idx = rng.integers(0, N)
        
        # Flip
        current_x[idx] *= -1
        
        # Compute new score
        new_score = _compute_c3_bound(current_x)
        
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        # Note: delta is in units of the score (max autocorr). 
        # If score is ~20, delta can be large.
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = current_x.copy()
        else:
            # Revert
            current_x[idx] *= -1
            
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # Optional: If stuck for too long, restart? 
        # For now, rely on SA.
        
    return best_x
# EVOLVE-BLOCK-END
