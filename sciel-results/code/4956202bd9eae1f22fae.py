"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _thue_morse(n):
    """Generate first n terms of Thue-Morse sequence (0/1)."""
    if n == 0:
        return np.array([], dtype=np.int8)
    seq = np.array([0], dtype=np.int8)
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]


def _autoconv_peak(signed):
    """Compute peak absolute value of autocorrelation of signed step function.
    
    The signed step function has values +/-1 on N intervals.
    The autoconvolution at lag k is sum_{i} f[i]*f[i+k].
    We want to minimize the maximum |autoconv| over all lags (excluding lag 0 typically,
    or including? The task says 'upper bound on C3' which relates to autocorrelation.
    
    For a signed step function with values s[i] in {-1, +1}, the autocorrelation is:
    R[k] = sum_{i=0}^{N-1-k} s[i] * s[i+k] for k >= 0
    
    The 'peak' usually refers to the maximum absolute value of R[k] for k != 0,
    or sometimes the maximum over all k. Let's assume we want to minimize
    max_{k=1}^{N-1} |R[k]|.
    
    Actually, the task mentions 'autoconvolution upper bound' and 'C3'.
    In the context of the third autocorrelation inequality (C3), we are likely
    minimizing the peak of the autocorrelation away from zero lag.
    """
    # Compute autocorrelation using FFT for efficiency
    # Pad to next power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = signed
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] is sum_{i=0}^{N-1-k} s[i]*s[i+k]
    # We exclude lag 0 (which is N) and look at lags 1..N-1
    peak = np.max(np.abs(acf[1:]))
    return peak


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial solution: Thue-Morse based signed sequence
    tm = _thue_morse(N)
    # Map 0 -> +1, 1 -> -1
    signed = 1 - 2 * tm
    signed = signed.astype(np.float64)
    
    best_signed = signed.copy()
    best_score = _autoconv_peak(best_signed)
    
    # Local search: flip individual bits to reduce peak autocorrelation
    # We'll use a simple greedy local search with multiple restarts
    
    def _local_search(signed_init, time_budget):
        s = signed_init.copy()
        current_peak = _autoconv_peak(s)
        n = len(s)
        
        # Precompute indices
        idx = np.arange(n)
        
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            # Try flipping each position
            # To speed up, we can shuffle the order
            order = np.random.permutation(n)
            for i in order:
                if (time.time() - start_time) > time_budget:
                    break
                # Flip bit i
                old_val = s[i]
                s[i] = -old_val
                
                new_peak = _autoconv_peak(s)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                    # Don't revert
                else:
                    s[i] = old_val
                    # Revert
            # If no improvement in a full pass, we could stop or continue
            # For simplicity, stop if no improvement
        return s, current_peak
    
    # Run local search with time budget
    time_budget = deadline - 5.0  # Leave 5s safety margin
    best_signed, best_score = _local_search(best_signed, time_budget)
    
    # If we have time, try a few random restarts
    if (time.time() - start_time) < time_budget - 10.0:
        for _ in range(3):  # Up to 3 restarts
            if (time.time() - start_time) > time_budget:
                break
            # Random restart
            s_rand = np.random.choice([-1.0, 1.0], size=N)
            s_rand, score_rand = _local_search(s_rand, time_budget)
            if score_rand < best_score:
                best_signed = s_rand
                best_score = score_rand
    
    return best_signed
# EVOLVE-BLOCK-END
