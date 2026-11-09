"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _generate_m_sequence(m):
    """Generate an m-sequence of length 2^m - 1 using LFSR."""
    # Use a known primitive polynomial for common m values
    # For m=8: x^8 + x^4 + x^3 + x^2 + 1 (0x1D in tap representation)
    # For m=9: x^9 + x^5 + 1
    # For m=10: x^10 + x^7 + 1
    # We'll use a general LFSR approach
    
    # For our purposes, we need a sequence close to length 400
    # 2^9 - 1 = 511, 2^8 - 1 = 255
    # Let's use m=9 (length 511) and take first 400, or use m=8 and repeat
    
    # Actually, let's generate a length-511 m-sequence and truncate to 400
    # Or better: generate multiple m-sequences and pick the best truncation
    
    if m == 8:
        taps = [4, 3, 2, 1, 0]  # x^8 + x^4 + x^3 + x^2 + 1
    elif m == 9:
        taps = [5, 0]  # x^9 + x^5 + 1
    elif m == 10:
        taps = [7, 0]  # x^10 + x^7 + 1
    else:
        # Fallback: use a simple LFSR
        taps = [m-1, 0]
    
    # Initialize LFSR state
    state = [1] * m
    seq = []
    length = (1 << m) - 1
    
    for i in range(length):
        seq.append(1 if state[-1] else -1)
        # Compute new bit
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        # Shift
        state = [new_bit] + state[:-1]
    
    return np.array(seq)


def _compute_autocorrelation(s):
    """Compute periodic autocorrelation for a ±1 sequence."""
    n = len(s)
    # Use FFT for efficient computation
    S = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(S * np.conj(S)).real[:n]
    acf /= n
    return acf


def _score(s):
    """Compute the upper bound C3 - we minimize max |R(k)| for k != 0."""
    acf = _compute_autocorrelation(s)
    # Exclude zero lag
    return np.max(np.abs(acf[1:]))


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Generate m-sequence of length 511 and truncate to 400
    m_seq = _generate_m_sequence(9)
    s = m_seq[:N].astype(float).copy()
    
    best_score = _score(s)
    best_s = s.copy()
    
    # Local search: flip individual elements to improve
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            # Try flipping element i
            s[i] *= -1
            new_score = _score(s)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_s = s.copy()
                improved = True
            else:
                s[i] *= -1  # revert
    
    # If we still have time, try a second pass with random restarts
    if time.time() < deadline - 10:
        # Try a few random perturbations of the best solution
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            s2 = best_s.copy()
            # Flip a few random elements
            for _ in range(10):
                idx = np.random.randint(0, N)
                s2[idx] *= -1
            # Local search from this perturbed state
            local_best = s2.copy()
            local_score = _score(s2)
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                s2[i] *= -1
                new_score = _score(s2)
                if new_score < local_score - 1e-12:
                    local_score = new_score
                    local_best = s2.copy()
                else:
                    s2[i] *= -1
            if local_score < best_score:
                best_score = local_score
                best_s = local_best
    
    return best_s
# EVOLVE-BLOCK-END
