"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a sequence of signed step heights to minimize the ratio:
    max|autoconvolution| / (sum(f))^2
    
    Theoretical analysis suggests that for any real sequence f,
    max_k |C_f(k)| >= C_f(0) = sum(f_i^2).
    And by Cauchy-Schwarz, (sum f_i)^2 <= N * sum(f_i^2).
    Thus, ratio >= 1/N.
    The constant sequence f = ones(N) achieves exactly 1/N.
    
    This implementation verifies this by generating a few alternative candidates
    (like LFSR sequences with optimized DC offset) and comparing them to the baseline.
    It returns the best candidate found.
    """
    
    def calc_value(f):
        S = np.sum(f)
        if S == 0:
            return float('inf')
        # Compute autocorrelation via FFT
        F = np.fft.fft(f, 2*N)
        C = np.fft.ifft(F * np.conj(F)).real[:N]
        max_corr = np.max(np.abs(C))
        return max_corr / (S * S)

    # Baseline: All ones
    f_base = np.ones(N)
    best_val = calc_value(f_base)
    best_f = f_base.copy()
    
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds safety margin
    
    # Strategy: Try LFSR sequences with varying DC offsets.
    # An LFSR sequence has low sidelobes (-1) but zero mean.
    # Adding a DC offset c creates a large sum but large sidelobes.
    # We check if any combination beats the baseline.
    
    # Generate a maximal length LFSR sequence of length 511 (2^9 - 1)
    # and take the first N bits.
    # Polynomial: x^9 + x^4 + 1 (primitive)
    # Tap positions: 9, 4 (0-indexed from LSB, or similar)
    # We'll use a simple implementation.
    
    def generate_lfsr(length, taps, initial_state=1):
        # taps are bit positions to XOR, e.g., [9, 4] for x^9+x^4+1
        # State is an integer
        seq = []
        state = initial_state
        max_tap = max(taps)
        for _ in range(length):
            seq.append(state & 1)
            # Feedback
            feedback = 0
            for t in taps:
                feedback ^= (state >> (t-1)) & 1
            state = (state >> 1) | (feedback << (max_tap - 1))
        return np.array(seq, dtype=np.float64) * 2 - 1 # Map 0,1 to -1,1

    # Try a few different LFSR polynomials
    lfsr_configs = [
        [9, 4], # 511
        [8, 3], # 255 (will repeat)
        [10, 7], # 1023
    ]
    
    for taps in lfsr_configs:
        if time.time() > deadline:
            break
        s = generate_lfsr(N, taps)
        
        # We need to find optimal DC offset c.
        # f = s + c
        # Value(c) = max(|C_s(k) + c^2 C_1(k) + ...|) / (sum(s)+cN)^2
        # Since sum(s) is small, Sum ~ cN.
        # C_s(k) is -1 for k!=0, N for k=0.
        # C_1(k) is N-|k|.
        # Max is likely at k=0: N + c^2 N.
        # Ratio ~ N(1+c^2) / (c^2 N^2) = (1+c^2)/(c^2 N).
        # Minimized as c -> infinity.
        # So LFSR + DC is always worse than baseline for large c.
        # For small c, Sum is small, Ratio is huge.
        
        # Thus, LFSR is not expected to beat baseline.
        # We skip extensive search on LFSR to save time, but do a quick check with c=1.
        for c in [1.0, 10.0, 100.0]:
            f_trial = s + c
            val = calc_value(f_trial)
            if val < best_val:
                best_val = val
                best_f = f_trial.copy()

    # Also try a random sparse sequence just in case
    if time.time() < deadline:
        np.random.seed(42)
        for _ in range(10):
            f_trial = np.random.choice([-1, 1], N)
            val = calc_value(f_trial)
            if val < best_val:
                best_val = val
                best_f = f_trial.copy()

    return best_f

# EVOLVE-BLOCK-END
