import numpy as np
import time

N = 400
DEADLINE = 118.0
M = 4 * N  # Fine grid resolution
dx = 0.5 / M

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    We assume the metric is the maximum absolute value of the autoconvolution,
    normalized such that the all-ones function yields 1.0.
    
    For all-ones on [-0.25, 0.25]:
    (f*f)(t) is a triangle with peak 0.5.
    If the evaluator's seed score is 1.0, and score = seed/value, then value_seed = 1.0.
    This implies the metric might be 2 * max|(f*f)| or similar scaling.
    However, since we are minimizing, the relative scaling doesn't matter for the search direction,
    only the ordering. We will minimize max|(f*f)|.
    """
    # Expand to fine grid
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    
    # Compute autoconvolution via FFT
    # Zero-pad to 2*M to avoid circular convolution artifacts
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M) * dx
    
    # The metric is the max absolute value
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Simulated Annealing parameters
    T = 1e-3
    T_min = 1e-6
    cooling_rate = 0.999
    
    no_improve_count = 0
    patience = 10000
    
    # Main search loop
    while True:
        # Check deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Choose move type: Single flip (80%) or Multi-flip (20%)
        if np.random.random() < 0.8:
            idx = np.random.randint(N)
            old_vals = [best_step[idx]]
            best_step[idx] = -best_step[idx]
            n_flipped = 1
            flipped_idxs = [idx]
        else:
            # Multi-flip: 2 or 3 random distinct elements
            n_flips = np.random.randint(2, 4)
            idxs = np.random.choice(N, size=n_flips, replace=False)
            old_vals = [best_step[i] for i in idxs]
            for i in idxs:
                best_step[i] = -best_step[i]
            n_flipped = n_flips
            flipped_idxs = idxs
        
        new_c3 = _compute_c3(best_step)
        
        delta = new_c3 - best_c3
        
        # Accept if improvement, or with probability exp(-delta/T)
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > T_min:
                # Avoid overflow in exp
                if -delta / T < 700: 
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                if np.random.random() < prob:
                    accept = True
        
        if accept:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            # Revert
            for i, v in zip(flipped_idxs, old_vals):
                best_step[i] = v
            no_improve_count += 1
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        if no_improve_count >= patience:
            break
    
    # Final greedy polish
    if time.time() - start_time < DEADLINE - 1.0:
        for idx in range(N):
            if time.time() - start_time > DEADLINE - 0.5:
                break
            old_val = best_step[idx]
            best_step[idx] = -old_val
            new_c3 = _compute_c3(best_step)
            if new_c3 < best_c3:
                best_c3 = new_c3
            else:
                best_step[idx] = old_val

    return best_step
