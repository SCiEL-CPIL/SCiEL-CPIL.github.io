"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds to spare before the 150s kill

def _autoconvolution(s):
    """Compute the full autoconvolution of the signed step sequence s.
    
    The autoconvolution at lag k is sum_{i} s[i] * s[i+k] (for valid ranges).
    We want to minimize the maximum absolute value of the autoconvolution
    excluding the zero-lag term (which is sum s[i]^2 = N for ±1 values).
    
    Actually, for a step function with N intervals, the autoconvolution 
    upper bound C3 relates to the maximum of the autocorrelation.
    Let's compute the full autocorrelation via FFT.
    """
    # Pad to avoid circular convolution artifacts
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft)[:N]
    # acf[0] = sum s^2 = N for ±1; lags 1..N-1 are the cross terms
    # The "third autocorrelation inequality" likely bounds |sum s[i] s[i+k]| for k != 0
    return acf

def _score(s):
    """Return the upper bound on C3 (lower is better).
    
    For a signed step function with ±1 values on N intervals, the autoconvolution
    at lag k is sum_{i=0}^{N-1-k} s[i]*s[i+k]. The maximum absolute value over
    non-zero lags gives the bound. We return max_{k=1}^{N-1} |acf[k]|.
    """
    acf = _autoconvolution(s)
    # Exclude lag 0
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _flip_block(s, start, length):
    """Flip a contiguous block of signs."""
    s[start:start+length] *= -1
    return s

def _compute_acf_fast(s):
    """Fast autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft)[:N]
    return acf

def _max_acf_excl_zero(acf):
    """Max absolute autocorrelation excluding lag 0."""
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def construct():
    start_time = time.time()
    
    # Initialize with random ±1 signs
    best_s = np.random.choice([-1, 1], size=N).astype(np.float64)
    best_acf = _compute_acf_fast(best_s)
    best_score = _max_acf_excl_zero(best_acf)
    
    # Try a few deterministic initializations
    for init_seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(init_seed)
        s_init = rng.choice([-1, 1], size=N).astype(np.float64)
        acf_init = _compute_acf_fast(s_init)
        score_init = _max_acf_excl_zero(acf_init)
        if score_init < best_score:
            best_s = s_init.copy()
            best_acf = acf_init
            best_score = score_init
    
    # Simulated annealing / local search
    # We'll use block flips of various lengths
    block_lengths = [1, 2, 3, 4, 5, 8, 16, 32, 64, 100, 200]
    
    current_s = best_s.copy()
    current_acf = best_acf.copy()
    current_score = best_score
    
    # Annealing schedule
    T_start = 1.0
    T_end = 0.01
    n_iter = 50000
    
    # Track elapsed time
    for it in range(n_iter):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Choose a random block length and position
        blen = block_lengths[np.random.randint(len(block_lengths))]
        if blen >= N:
            blen = N - 1
        start = np.random.randint(0, N - blen + 1)
        
        # Flip the block
        current_s[start:start+blen] *= -1
        
        # Compute new score
        new_acf = _compute_acf_fast(current_s)
        new_score = _max_acf_excl_zero(new_acf)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0:
            # Better
            current_score = new_score
            current_acf = new_acf
            if new_score < best_score:
                best_s = current_s.copy()
                best_acf = current_acf
                best_score = new_score
        else:
            # Worse: accept with probability
            T = T_start * (T_end / T_start) ** (it / n_iter)
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                current_acf = new_acf
            else:
                # Undo
                current_s[start:start+blen] *= -1
        
        # Periodically check time
        if it % 1000 == 0 and time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
    
    # Final polish: try single-bit flips and small blocks on the best solution
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        polish_s = best_s.copy()
        polish_acf = _compute_acf_fast(polish_s)
        polish_score = _max_acf_excl_zero(polish_acf)
        
        improved = True
        while improved and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
            improved = False
            for i in range(N):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                # Try flipping bit i
                polish_s[i] *= -1
                new_acf = _compute_acf_fast(polish_s)
                new_score = _max_acf_excl_zero(new_acf)
                if new_score < polish_score:
                    polish_score = new_score
                    polish_acf = new_acf
                    improved = True
                else:
                    polish_s[i] *= -1  # undo
        
        if polish_score < best_score:
            best_s = polish_s
            best_score = polish_score
    
    return best_s
# EVOLVE-BLOCK-END
