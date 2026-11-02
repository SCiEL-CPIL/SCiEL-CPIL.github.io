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


def construct():
    t_start = time.time()
    
    # Seed value: all ones
    seed_val = _score(np.ones(N, dtype=np.int8))
    
    best_seq = np.ones(N, dtype=np.int8)
    best_score = seed_val
    
    rng = np.random.default_rng(42)
    
    # Try several random initializations and local search
    n_attempts = 50
    max_flips_per_attempt = 5000
    
    for attempt in range(n_attempts):
        if time.time() - t_start > DEADLINE:
            break
        
        # Random ±1 initialization
        s = rng.choice([-1, 1], size=N).astype(np.int8)
        cur_score = _score(s)
        
        if cur_score < best_score:
            best_seq = s.copy()
            best_score = cur_score
        
        # Local search: flip individual bits
        improved = True
        flips = 0
        while improved and flips < max_flips_per_attempt:
            if time.time() - t_start > DEADLINE:
                break
            improved = False
            # Random order of positions to try
            order = rng.permutation(N)
            for i in order:
                if time.time() - t_start > DEADLINE:
                    break
                # Try flipping position i
                old_val = s[i]
                s[i] = -old_val
                new_score = _score(s)
                if new_score < cur_score:
                    cur_score = new_score
                    flips += 1
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_seq = s.copy()
                else:
                    s[i] = old_val
                    flips += 1
    
    # Final verification
    final_score = _score(best_seq)
    # The evaluator computes seed/value; we return the sequence
    return best_seq
