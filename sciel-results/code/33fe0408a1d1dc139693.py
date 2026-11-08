"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak_sidelobe(h):
    """Compute max |R[k]| for k != 0 using FFT-based linear autocorrelation."""
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)
    # Linear autocorrelation: R[0..n-1] are positive lags, R[nfft-n+1..nfft-1] negative lags
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft - n + 1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    # t(i) = parity of number of 1s in binary representation of i
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits set
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        bits[i] = 1 if cnt % 2 == 0 else -1
    return bits

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Initialize with Thue-Morse pattern
    h = _thue_morse(N).astype(np.float64)
    best_h = h.copy()
    best_score = _autocorr_peak_sidelobe(best_h)
    
    # Simulated annealing with block moves
    T_init = 50.0
    T_final = 0.1
    n_iterations = 50000
    block_sizes = [1, 2, 4, 8, 16, 32]
    
    rng = np.random.default_rng(42)
    
    for it in range(n_iterations):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Annealing schedule (geometric)
        T = T_init * (T_final / T_init) ** (it / n_iterations)
        
        # Choose a block size and position
        bs = block_sizes[rng.integers(0, len(block_sizes))]
        pos = rng.integers(0, N - bs + 1)
        
        # Flip the block
        h_new = h.copy()
        h_new[pos:pos+bs] *= -1
        
        # Compute new score
        score_new = _autocorr_peak_sidelobe(h_new)
        
        # Accept if better or with probability exp(-(delta)/T)
        delta = score_new - best_score
        if delta < 0 or (T > 0 and np.log(rng.random() + 1e-30) > -delta / T):
            h = h_new
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
    
    return best_h
# EVOLVE-BLOCK-END
