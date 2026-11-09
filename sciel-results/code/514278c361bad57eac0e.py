"""Program using LFSR-generated binary sequence for signed step function to minimize C3 upper bound."""
import numpy as np
from itertools import product

N = 400

# EVOLVE-BLOCK-START

def _is_primitive_poly(taps, m):
    """
    Check if the LFSR defined by taps and length m generates a maximal length sequence.
    taps: list of indices (0-indexed from LSB) where feedback occurs, excluding the leading 1.
    The polynomial is x^m + sum(x^tap for tap in taps) + 1 (if 0 in taps) ... 
    Standard form: p(x) = x^m + c_{m-1}x^{m-1} + ... + c_0.
    Here we define by the feedback polynomial.
    For m=9, taps [4, 0] means x^9 + x^4 + 1.
    """
    # Generate sequence
    state = [1] + [0] * (m - 1)
    seq = []
    for _ in range(2**m - 1):
        seq.append(state[0])
        # New bit is XOR of bits at tap positions
        # The taps are indices in the current state [b0, b1, ..., b_{m-1}]
        # The new bit enters at the front? 
        # Let's stick to the previous convention:
        # state = [b0, b1, ..., b_{m-1}]
        # new_bit = XOR(state[t] for t in taps)
        # state = [new_bit] + state[:-1]
        
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        state = [new_bit] + state[:-1]
        
    # Check if all bits are not zero (maximal length implies non-zero state doesn't repeat early)
    # Actually, if period is 2^m-1, it's maximal.
    # We just check if the sequence generated has length 2^m-1 and doesn't contain all zeros (impossible for LFSR starting non-zero)
    # A simpler check: does the state return to initial?
    # For the purpose of this search, we can just verify the period is 2^m-1 by checking the sequence doesn't repeat before.
    # But generating 511 or 1023 bits is fast.
    # We can just check if the sequence is not all same and period is correct.
    # Actually, if we generated 2^m-1 bits and the state is not back to start (which it shouldn't be for primitive), 
    # wait, after 2^m-1 steps, the state SHOULD be back to start.
    # So we just need to ensure it didn't cycle earlier.
    # Given the small size, we can just check if the number of unique states is 2^m-1.
    # Or simpler: just assume polynomials from a known list or generate and check period.
    
    # Let's do a quick period check:
    seen_states = set()
    state = [1] + [0] * (m - 1)
    for i in range(2**m):
        key = tuple(state)
        if key in seen_states:
            return i == 2**m - 1
        seen_states.add(key)
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        state = [new_bit] + state[:-1]
        
    return False

def _generate_lfsr_sequence(taps, m, length):
    state = [1] + [0] * (m - 1)
    seq = []
    for _ in range(length):
        seq.append(state[0])
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        state = [new_bit] + state[:-1]
    return np.array(seq, dtype=np.float64)

def construct():
    """
    Search for the best LFSR sequence (degree 9 or 10) and phase 
    that minimizes the maximum absolute aperiodic autocorrelation.
    """
    best_score = float('inf')
    best_heights = None
    
    # Search degrees 9 and 10
    for m in [9, 10]:
        period = 2**m - 1
        # Generate all possible tap combinations
        # Taps are subsets of {0, ..., m-2}
        # Note: The polynomial must be primitive.
        # We iterate over all non-empty subsets of {1, ..., m-1} plus optional 0?
        # Standard LFSR: x^m + sum c_i x^i. c_m=1.
        # The taps are the indices i where c_i=1.
        # We need at least one tap to be non-zero? 
        # Actually, the constant term c_0 must be 1 for it to be primitive (otherwise x is a factor).
        # So 0 must be in taps.
        
        # Iterate over all combinations of taps from 1 to m-1, and always include 0.
        for r in range(1, m): # number of non-zero taps besides x^m
            # Choose r taps from 1..m-1
            from itertools import combinations
            for comb in combinations(range(1, m), r):
                taps = list(comb) + [0]
                if _is_primitive_poly(taps, m):
                    # Generate full sequence
                    full_seq = _generate_lfsr_sequence(taps, m, period)
                    
                    # Search over phases
                    # We need to check all N possible windows? 
                    # Or just all period shifts?
                    # Since N < period, we check all period shifts.
                    # To save time, we can check a subset or use vectorization.
                    
                    # Optimization: 
                    # The autocorrelation of a shifted sequence is the same as the original 
                    # if we consider periodic autocorrelation. But we need APERIODIC.
                    # So we must check each phase.
                    
                    # We can speed this up by only checking a few promising phases 
                    # or using a heuristic. But with ~50 polynomials * ~500 phases * 2 degrees,
                    # and N=400, it's ~50,000 correlations.
                    # np.correlate is fast.
                    
                    for shift in range(period):
                        # Extract N bits starting at shift
                        # We need circular shift
                        idx = (np.arange(N) + shift) % period
                        seq = full_seq[idx]
                        
                        # Map to +/- 1
                        # full_seq is 0/1. Map 0->-1, 1->1
                        heights = 2.0 * seq - 1.0
                        
                        # Compute aperiodic autocorrelation
                        # R[k] = sum f[i] f[i+k]
                        # Use np.correlate
                        # np.correlate(a, v, mode='full')
                        # The output length is N + N - 1
                        # The peak is at index N-1
                        ac = np.correlate(heights, heights, mode='full')
                        
                        # Non-zero lags are all indices except the peak (N-1)
                        # We want max abs of these
                        # Exclude the peak
                        ac_non_peak = np.concatenate([ac[:N-1], ac[N:]])
                        
                        max_ac = np.max(np.abs(ac_non_peak))
                        
                        # Score: max_ac / N
                        score = max_ac / N
                        
                        if score < best_score:
                            best_score = score
                            best_heights = heights.copy()
                            
    # If no best found (should not happen), fallback to random
    if best_heights is None:
        best_heights = np.ones(N)
        
    return best_heights

# EVOLVE-BLOCK-END
