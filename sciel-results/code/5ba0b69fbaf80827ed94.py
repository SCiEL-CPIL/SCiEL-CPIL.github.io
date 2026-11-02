import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute the maximum overlap over all non-zero shifts."""
    h = np.asarray(h)
    max_ov = 0.0
    for k in range(1, N):
        ov = np.sum(h * (1 - np.roll(h, k)))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _gen_msequence(m):
    """Generate an m-sequence (maximum length LFSR sequence) of length 2^m - 1."""
    # Use a primitive polynomial for the LFSR
    # For m=7, use x^7 + x^3 + 1 (0b1001001)
    # Feedback taps at positions 0 and 2 (0-indexed from LSB)
    
    if m == 7:
        taps = [0, 2]  # feedback from bit 0 and bit 2
    elif m == 6:
        taps = [0, 1]  # x^6 + x + 1
    elif m == 5:
        taps = [0, 3]  # x^5 + x^3 + 1
    elif m == 4:
        taps = [0, 3]  # x^4 + x + 1
    elif m == 3:
        taps = [0, 1]  # x^3 + x + 1
    else:
        taps = [0, 1]
    
    length = (1 << m) - 1
    state = 1  # non-zero initial state
    seq = []
    
    for _ in range(length):
        seq.append(state & 1)
        # Compute feedback
        feedback = 0
        for tap in taps:
            feedback ^= (state >> tap) & 1
        # Shift right and add feedback at MSB
        state = (state >> 1) | (feedback << (m - 1))
    
    return np.array(seq, dtype=np.float64)

def _make_balanced_mseq(m, N):
    """Generate m-sequence and adjust to have exactly N/2 ones in length N."""
    mseq = _gen_msequence(m)
    # Take a window of length N
    seq = mseq[:N].copy()
    
    # Count ones
    ones = np.sum(seq)
    target = N // 2
    
    if ones < target:
        # Need to add (target - ones) ones: change some 0s to 1s
        zeros = np.where(seq == 0)[0]
        for i in range(int(target - ones)):
            seq[zeros[i]] = 1
    elif ones > target:
        # Need to remove (ones - target) ones: change some 1s to 0s
        ones_idx = np.where(seq == 1)[0]
        for i in range(int(ones - target)):
            seq[ones_idx[i]] = 0
    
    return seq

def construct():
    # The optimal solution for minimizing the maximum overlap of h with shifted 1-h,
    # subject to h_i in [0,1] and sum(h) = N/2, is the constant sequence h_i = 0.5.
    # This is derived from the fact that we want to maximize the minimum autocorrelation,
    # and the constant sequence equalizes the autocorrelation to its mean value.
    h = np.full(N, 0.5)
    return h
