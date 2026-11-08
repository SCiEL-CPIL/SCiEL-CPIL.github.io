import numpy as np
import time
import math

N = 400
DEADLINE = 145.0
M = 2 * N  # Fine grid resolution (2x N is sufficient for step function)
dx = 0.5 / M

# Precompute the index mapping from fine grid to coarse intervals
_FINE_TO_COARSE = np.clip(np.arange(M) * N // M, 0, N - 1)

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    f_fine = signed_step[_FINE_TO_COARSE].astype(np.float64)
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Try a few random initializations to find a better starting point
    for trial in range(8):
        if time.time() - start_time > DEADLINE - 20.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Simulated annealing
    current_step = best_step.copy()
    current_c3 = best_c3
    
    # SA parameters
    T0 = 0.05  # Initial temperature (scale relative to typical C3 delta)
    T_min = 1e-6
    cooling_rate = 0.9995  # Geometric cooling
    T = T0
    
    elapsed_limit = DEADLINE - 3.0  # Leave 3s for final polish
    sa_start = time.time()
    
    # Estimate typical delta scale from first few moves
    # We'll adapt T based on observed deltas
    
    for iteration in range(200000):
        if time.time() - sa_start > elapsed_limit - sa_start:
            break
        if time.time() - start_time > DEADLINE - 3.0:
            break
        
        # Choose move: single flip (primary) or occasional double flip
        if np.random.random() < 0.85:
            idx = np.random.randint(N)
            idxs = [idx]
        else:
            idxs = list(np.random.choice(N, size=2, replace=False))
        
        # Apply move
        old_vals = current_step[idxs].copy()
        for idx in idxs:
            current_step[idx] = -current_step[idx]
        
        new_c3 = _compute_c3(current_step)
        delta = new_c3 - current_c3
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvements
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Metropolis acceptance: accept with probability exp(-delta/T)
            if T > 1e-10:
                accept_prob = math.exp(-delta / T)
                if np.random.random() < accept_prob:
                    current_c3 = new_c3
                else:
                    # Reject: revert
                    for i, idx in enumerate(idxs):
                        current_step[idx] = old_vals[i]
            else:
                # Reject: revert
                for i, idx in enumerate(idxs):
                    current_step[idx] = old_vals[i]
        
        # Cool down
        T *= cooling_rate
    
    # Final greedy polish: try each single flip in sequence
    if time.time() - start_time < DEADLINE - 1.0:
        polish_step = best_step.copy()
        polish_c3 = best_c3
        for idx in range(N):
            if time.time() - start_time > DEADLINE - 0.5:
                break
            old_val = polish_step[idx]
            polish_step[idx] = -old_val
            new_c3 = _compute_c3(polish_step)
            if new_c3 < polish_c3:
                polish_c3 = new_c3
            else:
                polish_step[idx] = old_val
        best_step = polish_step
        best_c3 = polish_c3
    
    return best_step
