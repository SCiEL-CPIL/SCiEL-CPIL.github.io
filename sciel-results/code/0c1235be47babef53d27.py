"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(seq, exclude_lag0=True):
    """Compute max absolute autoconvolution, optionally excluding lag 0."""
    n = len(seq)
    # Full autoconvolution via FFT for speed
    # Pad to avoid circular convolution
    size = 2 * n
    fft_len = 1
    while fft_len < size:
        fft_len *= 2
    f = np.fft.rfft(seq, n=fft_len)
    ac = np.fft.irfft(f * np.conj(f), n=fft_len)[:size]
    # ac[k] for k=0..n-1 is the linear autoconv at lag k
    # ac[n-k] for k=1..n-1 corresponds to negative lags
    if exclude_lag0:
        # Take max over all lags except 0
        # ac[0] is lag 0, ac[1:n] are positive lags, ac[n:n-1:-1] are negative lags
        vals = np.concatenate([ac[1:n], ac[n:n-1:-1]])
        return np.max(np.abs(vals))
    else:
        return np.max(np.abs(ac[:n]))

def _score(seq):
    return _autoconv_max_abs(seq, exclude_lag0=True)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    np.random.seed(42)
    random.seed(42)
    
    # Start with a random ±1 sequence
    seq = np.random.choice([-1, 1], size=N).astype(np.float64)
    best_seq = seq.copy()
    best_score = _score(seq)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Pick a random position to flip
        i = np.random.randint(0, N)
        old_val = seq[i]
        seq[i] = -old_val
        
        new_score = _score(seq)
        
        if new_score < best_score:
            best_score = new_score
            best_seq = seq.copy()
        else:
            # Accept with probability exp(-(new_score - best_score)/T)
            if new_score <= best_score:
                pass  # already accepted
            else:
                delta = new_score - best_score
                if np.random.random() < np.exp(-delta / T):
                    pass  # accept
                else:
                    seq[i] = old_val  # reject
        
        T *= cooling
    
    return best_seq
# EVOLVE-BLOCK-END
