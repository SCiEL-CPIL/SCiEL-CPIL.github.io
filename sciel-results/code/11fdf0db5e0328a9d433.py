"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autoconv(s):
    """Compute autoconvolution values at integer lags for sign sequence s."""
    n = len(s)
    # s is array of ±1
    # autoconv at lag k is sum_{j} s[j]*s[j+k] for valid j
    # Use FFT for efficiency
    # Zero-pad to 2n to avoid circular convolution
    nfft = 1
    while nfft < 2*n:
        nfft *= 2
    S = np.zeros(nfft)
    S[:n] = s
    F = np.fft.fft(S)
    acf = np.fft.ifft(F * np.conj(F)).real[:n]
    return acf

def _score(s):
    """Compute the score (upper bound on C3). Lower is better."""
    acf = _compute_autoconv(s)
    # The autoconvolution at lag 0 is always n (sum of s[j]*s[j] = n)
    # We want to minimize some measure of the autoconvolution
    # Let's try the maximum absolute value excluding lag 0
    # Or perhaps the L2 norm of the sidelobes
    # Let's use the max of |acf[k]| for k != 0
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Initialize with random signs
    np.random.seed(42)
    s = np.random.choice([-1, 1], size=N)
    best_s = s.copy()
    best_score = _score(s)
    
    # Simulated annealing with block flips
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick a random block to flip
        block_size = np.random.randint(4, 33)  # 4 to 32
        if block_size > N:
            block_size = N
        start = np.random.randint(0, N - block_size + 1)
        
        # Flip the block
        s[start:start+block_size] *= -1
        
        new_score = _score(s)
        delta = new_score - best_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            best_s = s.copy()
            best_score = new_score
        else:
            # Undo
            s[start:start+block_size] *= -1
        
        T *= cooling
    
    return best_s
# EVOLVE-BLOCK-END
