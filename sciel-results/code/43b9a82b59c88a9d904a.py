"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(signed_steps):
    """Compute the upper bound on C3 from a signed step function."""
    n = len(signed_steps)
    # Autoconvolution: sum of f[i]*f[j] for i+j = k
    # Using FFT for efficiency
    # Pad to avoid circular convolution
    size = 2 * n
    f_padded = np.zeros(size)
    f_padded[:n] = signed_steps
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.rfft(f_padded)
    conv = np.fft.irfft(f_fft * f_fft, n=size)
    
    # The autoconvolution upper bound relates to the peak of |conv|
    # For the C3 bound, we look at the maximum absolute value
    c3_bound = np.max(np.abs(conv))
    return c3_bound

def _logistic_map_sequence(n, r, x0=0.5):
    """Generate a sequence using the logistic map in chaotic regime."""
    seq = np.zeros(n)
    x = x0
    for i in range(n):
        x = r * x * (1.0 - x)
        seq[i] = 1.0 if x > 0.5 else -1.0
    return seq

def _local_search(signed_steps, max_iter=5000, time_limit=100):
    """Local search over sign flips to minimize autoconvolution peak."""
    n = len(signed_steps)
    current = signed_steps.copy()
    current_bound = _compute_c3_bound(current)
    
    start_time = time.time()
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try random sign flips
        i = np.random.randint(n)
        new = current.copy()
        new[i] *= -1
        new_bound = _compute_c3_bound(new)
        
        if new_bound < current_bound:
            current = new
            current_bound = new_bound
    
    return current, current_bound

def construct():
    """Construct a signed step function to minimize C3 upper bound."""
    deadline = time.time() + 115  # 115s safety margin
    
    # Strategy 1: Try multiple logistic map parameters
    best_seq = None
    best_bound = np.inf
    
    # Chaotic regime: r in (3.57, 4.0]
    r_values = [3.7, 3.8, 3.9, 3.95, 3.99, 4.0]
    x0_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for r in r_values:
        if time.time() > deadline - 30:
            break
        for x0 in x0_values:
            if time.time() > deadline - 30:
                break
            seq = _logistic_map_sequence(N, r, x0)
            bound = _compute_c3_bound(seq)
            if bound < best_bound:
                best_bound = bound
                best_seq = seq.copy()
    
    # Strategy 2: Local search refinement
    if best_seq is not None:
        refined, refined_bound = _local_search(best_seq, max_iter=2000, time_limit=60)
        if refined_bound < best_bound:
            best_seq = refined
            best_bound = refined_bound
    
    # Strategy 3: Multi-restart local search from different initializations
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        r = np.random.uniform(3.7, 4.0)
        x0 = np.random.uniform(0.1, 0.9)
        seq = _logistic_map_sequence(N, r, x0)
        refined, refined_bound = _local_search(seq, max_iter=1000, time_limit=15)
        if refined_bound < best_bound:
            best_seq = refined
            best_bound = refined_bound
    
    if best_seq is None:
        # Fallback: Thue-Morse sequence
        best_seq = np.ones(N)
        for i in range(N):
            # Count bits in i
            best_seq[i] = -1.0 if bin(i).count('1') % 2 else 1.0
    
    return best_seq
# EVOLVE-BLOCK-END
