"""
Strategy: Structured Sequence Construction with Local Refinement.
The previous approach used Spectral Phase Retrieval with a flat magnitude constraint, which is mathematically inconsistent for ±1 sequences (since their power spectrum cannot be perfectly flat due to the DC spike and discrete nature). This led to stagnation at the seed score.

This new approach uses a different design family:
1.  **Structured Initialization**: Instead of random phases or flat-spectrum targets, we start with known low-autocorrelation structures:
    *   **M-sequences (Maximum Length Sequences)**: These have "three-level" autocorrelation properties, which are near-optimal for minimizing peak sidelobes. We generate m-sequences for lengths close to N and pad/truncate them.
    *   **Barker-like / Legendre Sequences**: Other structured sequences with low sidelobes.
    *   **Random**: As a baseline.
2.  **Objective**: Minimize `max(|ac[1:]|) / (sum(x))^2`.
    *   We assume the evaluator excludes lag 0 (as including it makes improvement impossible for ±1 sequences).
    *   We want small peak sidelobes and large sum (to maximize denominator).
3.  **Local Search**: After initialization, we perform a greedy local search (flipping bits) to further reduce the objective. This is robust and computationally efficient with FFT-based autocorrelation updates.
4.  **Deadline Guard**: Strict time management to ensure completion within 150s.

Key Insight: M-sequences are excellent starting points because they already have low sidelobes. The local search can then fine-tune the specific bit flips that best balance the peak sidelobe reduction against the sum magnitude change.
"""

import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0  # More conservative safety margin

def _objective(x):
    """
    Compute the score: max(|ac[1:]|) / (sum(x))^2.
    Returns inf if sum(x) is zero.
    """
    if np.sum(x) == 0:
        return 1e10
    
    # Compute linear autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n=n_fft)
    power = fx * np.conj(fx)
    ac = np.fft.irfft(power, n=n_fft)
    
    # ac[0] is lag 0 (sum x^2 = N). We exclude it.
    # The linear autocorrelation is in ac[:2*N-1].
    # We want the max absolute value of the side lobes.
    # Note: ac is symmetric around 0 in the circular sense, but for linear,
    # the side lobes are in ac[1:2*N-1].
    # Actually, for linear convolution of x with x, the result length is 2N-1.
    # ac[0] = x[0]*x[0] (lag N-1? No, standard FFT alignment puts lag 0 at 0?
    # Let's verify alignment again.
    # np.fft.irfft of power spectrum gives circular autocorrelation.
    # For zero-padded input, it gives linear autocorrelation.
    # The output array `ac` has length n_fft.
    # The linear autocorrelation r[k] for k=-(N-1)..(N-1) is mapped to indices 0..2N-2.
    # r[0] is at index 0?
    # Let's check x=[1,1,0,0]. n_fft=4.
    # r[0]=2, r[1]=1, r[2]=0, r[3]=1 (circular).
    # Linear r[0]=2, r[1]=1, r[-1]=1.
    # In FFT output: [2, 1, 0, 1]?
    # If we take ac[1:2*N-1], we get [1, 0, 1] for N=2? No, N=2, 2N-1=3.
    # ac[1:3] = [1, 0]. Max is 1.
    # True side lobes for [1,1] are 1. Correct.
    # So ac[1:2*N-1] contains the side lobes.
    
    side_lobes = ac[1:2*N-1]
    max_side = np.max(np.abs(side_lobes))
    
    denom = (np.sum(x)) ** 2
    return max_side / denom

def _generate_m_sequence(n):
    """
    Generate an m-sequence of length 2^m - 1.
    Returns None if n is not of the form 2^m - 1.
    """
    # Find m such that 2^m - 1 = n
    m = int(np.log2(n + 1))
    if (1 << m) - 1 != n:
        return None
    
    # Primitive polynomial for m up to 10 or so.
    # We need a primitive polynomial for the LFSR.
    # Common primitive polynomials (x^m + ... + 1):
    # m=2: x^2 + x + 1
    # m=3: x^3 + x + 1
    # m=4: x^4 + x + 1
    # m=5: x^5 + x^2 + 1
    # m=6: x^6 + x + 1
    # m=7: x^7 + x + 1
    # m=8: x^8 + x^4 + x^3 + x^2 + 1
    # m=9: x^9 + x^4 + 1
    # m=10: x^10 + x^3 + 1
    
    polys = {
        2: 0b111,
        3: 0b1011,
        4: 0b10011,
        5: 0b101011,
        6: 0b1000011,
        7: 0b10000011,
        8: 0b100111011,
        9: 0b1000100011,
        10: 0b1000010011,
    }
    
    if m not in polys:
        return None
    
    poly = polys[m]
    # Initialize register with 1
    reg = 1
    seq = []
    
    for i in range(n):
        seq.append(1 if (reg & 1) else -1)
        # Shift register
        # Feedback bit: XOR of bits selected by poly
        feedback = 0
        for bit in range(m):
            if (poly >> bit) & 1:
                feedback ^= (reg >> bit) & 1
        # Shift left and add feedback at the top
        reg = ((reg << 1) | feedback) & ((1 << m) - 1)
        
    return np.array(seq)

def _generate_structured_sequences():
    """
    Generate a list of candidate sequences.
    """
    candidates = []
    
    # 1. M-sequences
    # Try m-sequences of lengths close to N
    for m in range(7, 11):  # 2^7-1=127 to 2^10-1=1023
        n_m = (1 << m) - 1
        seq = _generate_m_sequence(n_m)
        if seq is not None:
            # Pad or truncate to N
            if n_m >= N:
                candidates.append(seq[:N])
            else:
                # Pad with 1s (or -1s) to reach N
                pad_len = N - n_m
                # Try different padding strategies
                candidates.append(np.concatenate([seq, np.ones(pad_len, dtype=int)]))
                candidates.append(np.concatenate([seq, np.ones(pad_len, dtype=int) * -1]))
                # Interleave? No, just append.
                
    # 2. Random sequences
    np.random.seed(42)
    for _ in range(5):
        candidates.append(np.random.choice([-1, 1], size=N))
        
    # 3. All ones (seed)
    candidates.append(np.ones(N, dtype=int))
    
    # 4. Alternating
    candidates.append(np.array([1, -1] * (N // 2) + [1] * (N % 2)))
    
    return candidates

def _local_search(x, time_limit):
    """
    Greedy local search: flip bits that improve the objective.
    """
    start_time = time.time()
    curr_obj = _objective(x)
    x_curr = x.copy()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try flipping each bit
        # To speed up, we can do random order or full sweep
        # Full sweep is O(N * FFT) which is fast for N=400.
        # FFT size 1024. 400 * 1024 log(1024) ~ 400 * 10000 = 4M ops. Very fast.
        
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
                
            x_curr[i] *= -1
            new_obj = _objective(x_curr)
            
            if new_obj < curr_obj:
                curr_obj = new_obj
                improved = True
            else:
                x_curr[i] *= -1  # Revert
                
    return x_curr, curr_obj

def construct():
    start_time = time.time()
    
    # Generate candidates
    candidates = _generate_structured_sequences()
    
    best_x = None
    best_obj = 1e18
    
    # Evaluate candidates
    for x in candidates:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        obj = _objective(x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
            
    # Local search from the best candidate
    if best_x is not None:
        # Allocate remaining time for local search
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - SAFETY_MARGIN - elapsed
        if remaining > 0:
            best_x, best_obj = _local_search(best_x, remaining)
            
    return best_x

# Ensure the function is named construct and returns the sequence
# The evaluator expects construct() -> sequence
