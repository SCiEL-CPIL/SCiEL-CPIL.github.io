"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE = 145  # safety margin before 150s kill


def _max_autocorr_abs(s):
    """Compute max |autocorr| over non-zero lags for a length-N sequence.
    
    Returns the maximum of |r[k]| for k=1..N-1 where r is the linear autocorrelation.
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # Linear autocorrelation: acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k]
    # Normalize: divide by N
    acf = acf / N
    return float(np.max(np.abs(acf[1:])))


def _score(s):
    """Upper bound on C3; lower is better."""
    return _max_autocorr_abs(s)


def _chirp_sequence(N):
    """Generate a real chirp-like sequence."""
    n = np.arange(N)
    phase = np.pi * n * n / N
    return np.sign(np.cos(phase)).astype(np.int8)


def _block_flip(current_seq, rng, block_size):
    """Perform a block flip: flip a contiguous block of signs."""
    start = rng.integers(N - block_size + 1)
    end = start + block_size
    current_seq[start:end] = -current_seq[start:end]


def _evaluate_block_flip(current_seq, rng, block_size, T):
    """Try a block flip and accept/reject based on simulated annealing."""
    old_seq = current_seq.copy()
    old_score = _score(current_seq)
    
    _block_flip(current_seq, rng, block_size)
    new_score = _score(current_seq)
    
    delta = new_score - old_score
    if delta < 0 or rng.random() < np.exp(-delta / T):
        return True, new_score
    else:
        current_seq[:] = old_seq
        return False, old_score


def _multi_resolution_initialization(N, rng):
    """Initialize sequence using multi-resolution approach."""
    # Start with a chirp
    seq = _chirp_sequence(N).astype(np.int8)
    
    # Apply some random perturbations at different scales
    for scale in [1, 2, 4, 8, 16, 32]:
        for _ in range(10):
            start = rng.integers(0, N - scale + 1)
            end = start + scale
            # Flip with some probability
            if rng.random() < 0.5:
                seq[start:end] = -seq[start:end]
    
    return seq


def construct():
    t_start = time.time()
    
    # Seed value: all ones
    seed_val = _score(np.ones(N, dtype=np.int8))
    
    # Initialize with multi-resolution approach
    best_seq = _multi_resolution_initialization(N, np.random.default_rng(42))
    best_score = _score(best_seq)
    
    # Also check chirp and ones
    chirp_seq = _chirp_sequence(N)
    chirp_score = _score(chirp_seq)
    if chirp_score < best_score:
        best_score = chirp_score
        best_seq = chirp_seq.copy()
    
    ones_score = seed_val
    if ones_score < best_score:
        best_seq = np.ones(N, dtype=np.int8)
        best_score = ones_score
        
    rng = np.random.default_rng(42)
    
    # Simulated Annealing with block flips
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-6
    
    current_seq = best_seq.copy()
    current_score = best_score
    
    # Different block sizes for exploration
    block_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    while T > min_T:
        if time.time() - t_start > DEADLINE:
            break
        
        # Choose block size based on temperature
        if T > 0.5:
            # Higher temperature: larger blocks for exploration
            block_size = block_sizes[rng.integers(3, len(block_sizes))]
        elif T > 0.1:
            # Medium temperature: medium blocks
            block_size = block_sizes[rng.integers(1, 5)]
        else:
            # Low temperature: small blocks for exploitation
            block_size = block_sizes[rng.integers(0, 3)]
        
        accepted, new_score = _evaluate_block_flip(current_seq, rng, block_size, T)
        
        if accepted:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_seq = current_seq.copy()
        
        T *= cooling_rate
        
        # Occasionally try a completely random restart if stuck
        if rng.random() < 0.001:
            restart_seq = _multi_resolution_initialization(N, rng)
            restart_score = _score(restart_seq)
            if restart_score < best_score:
                best_score = restart_score
                best_seq = restart_seq.copy()
                current_seq = restart_seq.copy()
                current_score = restart_score
                T = 0.5  # Reset temperature
    
    return best_seq
