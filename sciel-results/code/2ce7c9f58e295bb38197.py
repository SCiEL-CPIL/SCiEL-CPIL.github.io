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
    # Scale to fit in [0, 2*pi] or similar
    # For ±1, we can use sign of cos
    phase = np.pi * n * n / N
    return np.sign(np.cos(phase)).astype(np.int8)

def _m_sequence(N):
    """Generate a pseudo-random M-sequence like sequence.
    Since 400 is not 2^m-1, we use a random LFSR-like generation or just high-quality random.
    """
    # Use a fixed seed for reproducibility but different from random
    rng = np.random.default_rng(123)
    return rng.choice([-1, 1], size=N).astype(np.int8)

def _greedy_improve(s, max_iter=100):
    """Perform greedy bit flips to improve the score."""
    cur_score = _score(s)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        # Try flipping each bit
        for i in range(N):
            old_val = s[i]
            s[i] = -old_val
            new_score = _score(s)
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                s[i] = old_val
    return s, cur_score

def construct():
    t_start = time.time()
    
    # Generate diverse initial candidates
    candidates = [
        np.ones(N, dtype=np.int8),
        _chirp_sequence(N),
        _m_sequence(N),
        np.random.default_rng(42).choice([-1, 1], size=N).astype(np.int8)
    ]
    
    # Evaluate all candidates and pick the best initial one
    scores = [_score(c) for c in candidates]
    best_idx = np.argmin(scores)
    best_seq = candidates[best_idx].copy()
    best_score = scores[best_idx]
    
    # Phase 1: Greedy improvement on the best candidate
    best_seq, best_score = _greedy_improve(best_seq, max_iter=50)
    
    # Phase 2: Simulated Annealing to escape local minima
    # Use a faster cooling schedule
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    
    s = best_seq.copy()
    cur_score = best_score
    
    rng = np.random.default_rng(1234)
    
    while time.time() - t_start < DEADLINE:
        if T < T_min:
            # If we cooled down too much, restart with a new random sequence
            # or just break if we've been stuck
            if cur_score >= best_score:
                # Try a few random restarts if time allows
                if time.time() - t_start < DEADLINE - 10:
                    s = rng.choice([-1, 1], size=N).astype(np.int8)
                    s, cur_score = _greedy_improve(s, max_iter=10)
                    if cur_score < best_score:
                        best_score = cur_score
                        best_seq = s.copy()
                    T = 0.05
                    continue
                else:
                    break
        
        # Perform one step of SA
        # Flip a single bit
        i = rng.integers(0, N)
        
        # Save old value
        old_val = s[i]
        
        # Flip
        s[i] = -old_val
        
        new_score = _score(s)
        
        # Acceptance probability
        delta = new_score - cur_score
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_seq = s.copy()
        else:
            # Revert
            s[i] = old_val
            
        T *= cooling_rate
        
    # Final verification
    final_score = _score(best_seq)
    return best_seq
