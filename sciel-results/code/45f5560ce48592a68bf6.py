import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid using vectorized repeat
    f_fine = np.repeat(signed_step, M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Try to improve using local search with multi-flip moves
    np.random.seed(42)
    
    no_improve_count = 0
    patience = 5000
    
    while time.time() - start_time < DEADLINE - 5.0:
        # Choose number of flips: mostly 1, sometimes 2 or 3
        r = np.random.random()
        if r < 0.7:
            n_flips = 1
        elif r < 0.9:
            n_flips = 2
        else:
            n_flips = 3
        
        # Select random indices to flip (without replacement)
        idxs = np.random.choice(N, size=n_flips, replace=False)
        old_vals = best_step[idxs].copy()
        best_step[idxs] = -best_step[idxs]
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            best_step[idxs] = old_vals
            no_improve_count += 1
        
        if no_improve_count >= patience:
            # Reset patience with a larger random perturbation to escape local minima
            n_perturb = np.random.randint(5, 20)
            idxs = np.random.choice(N, size=n_perturb, replace=False)
            best_step[idxs] = -best_step[idxs]
            perturbed_c3 = _compute_c3(best_step)
            if perturbed_c3 < best_c3:
                best_c3 = perturbed_c3
            else:
                best_step[idxs] = -best_step[idxs]  # revert
            no_improve_count = 0
    
    return best_step
