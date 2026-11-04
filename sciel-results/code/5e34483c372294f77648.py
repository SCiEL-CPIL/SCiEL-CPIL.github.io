"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0-indexed, values 0/1)."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def _autoconv_max_abs(s):
    """Compute max |R[k]| for autoconvolution of sequence s (values +/-1)."""
    # s is 1D array of +/-1, length N
    # R[k] = sum_{i=0}^{N-1-|k|} s[i] * s[i+|k|] for k >= 0
    # By symmetry R[-k] = R[k]
    N = len(s)
    # Use FFT for efficiency
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * N:
        size *= 2
    F = np.fft.rfft(s.astype(np.float64), size)
    power = F * np.conj(F)
    R = np.fft.irfft(power, size)
    # R[0:N] gives linear autoconvolution R[0], R[1], ..., R[N-1]
    # R[k] for k=0..N-1
    # The full autoconvolution has R[k] for k = -(N-1)..(N-1)
    # By symmetry, max |R[k]| = max(R[0:N]) since R is symmetric and non-negative? 
    # No! R[k] can be negative for signed sequences.
    # Actually for autoconvolution: R[k] = sum s[i]*s[i+k], and R[-k] = R[k]
    # So we just need max over k=0..N-1 of |R[k]|
    return np.max(np.abs(R[:N]))

def _local_search(s, max_time=110.0):
    """Local search: flip bits to reduce max autoconvolution."""
    s = s.copy()
    current_val = _autoconv_max_abs(s)
    start_time = time.time()
    N = len(s)
    
    # Precompute indices
    indices = np.arange(N)
    
    # Try single-bit flips
    # For efficiency, compute the effect of flipping bit j on R[k]
    # Flipping s[j] changes R[k] by -2*s[j]*s[j+k] for k where both indices valid
    # and by -2*s[j]*s[j-k] for negative k (but by symmetry we handle k>=0)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        improved = False
        # Try random order of flips
        order = np.random.permutation(N)
        
        for j in order:
            if time.time() - start_time > max_time:
                break
            
            old_val = current_val
            s[j] = -s[j]
            new_val = _autoconv_max_abs(s)
            if new_val < old_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                s[j] = -s[j]  # revert
        
        if not improved:
            break
    
    return s, current_val

def _pairwise_swap_search(s, max_time=110.0):
    """Try swapping pairs of bits to improve."""
    s = s.copy()
    current_val = _autoconv_max_abs(s)
    start_time = time.time()
    N = len(s)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        improved = False
        # Try random pairs
        for _ in range(200):
            if time.time() - start_time > max_time:
                break
            i, j = np.random.randint(0, N, size=2)
            if i == j:
                continue
            s[i], s[j] = s[j], s[i]
            new_val = _autoconv_max_abs(s)
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                s[i], s[j] = s[j], s[i]  # revert
        
        if not improved:
            break
    
    return s, current_val

def construct():
    """Construct signed step function minimizing autoconvolution L-infinity norm."""
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    # Initialize with Thue-Morse (values +/-1)
    tm = _thue_morse(N)
    s = np.where(tm == 0, 1, -1).astype(np.float64)
    
    # Multi-restart local search
    best_s = s.copy()
    best_val = _autoconv_max_abs(s)
    
    # First local search from Thue-Morse
    s_ls, val_ls = _local_search(s, max_time=DEADLINE - (time.time() - start_time) - 5.0)
    if val_ls < best_val:
        best_s = s_ls
        best_val = val_ls
    
    # Pairwise swap search
    if time.time() - start_time < DEADLINE - 5.0:
        s_ps, val_ps = _pairwise_swap_search(best_s, max_time=DEADLINE - (time.time() - start_time) - 3.0)
        if val_ps < best_val:
            best_s = s_ps
            best_val = val_ps
    
    # Try a few more random restarts if time permits
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10.0:
        np.random.seed(42)
        for restart in range(3):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            # Random initialization
            s_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
            s_ls2, val_ls2 = _local_search(s_rand, max_time=min(remaining - 5.0, 20.0))
            if val_ls2 < best_val:
                best_s = s_ls2
                best_val = val_ls2
            remaining = DEADLINE - (time.time() - start_time)
            if remaining < 5.0:
                break
    
    # Final answer
    return best_s

# EVOLVE-BLOCK-END
