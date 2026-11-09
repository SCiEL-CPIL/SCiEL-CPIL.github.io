"""Program using LFSR-generated binary sequence for signed step function to minimize C3 upper bound."""
import numpy as np

N = 400

# EVOLVE-BLOCK-START

# List of primitive polynomials (m, taps) where taps are indices of non-zero coefficients
# excluding the leading x^m term. Index 0 corresponds to x^0 (constant term).
# For a primitive polynomial, the constant term must be 1, so 0 is always in taps.
PRIMITIVE_POLYS = [
    (9, [0, 4]),
    (9, [0, 5]),
    (9, [0, 6, 7]),
    (9, [0, 3, 4]),
    (9, [0, 5, 6]),
    (9, [0, 1, 8]),
    (9, [0, 2, 8]),
    (9, [0, 3, 7]),
    (9, [0, 4, 7]),
    (9, [0, 6, 8]),
    (9, [0, 1, 4, 7]),
    (9, [0, 2, 5, 7]),
    (9, [0, 3, 5, 7]),
    (9, [0, 4, 6, 7]),
    (10, [0, 3]),
    (10, [0, 7]),
    (10, [0, 3, 8]),
    (10, [0, 4, 9]),
    (10, [0, 5, 9]),
    (10, [0, 6, 9]),
    (10, [0, 1, 9]),
    (10, [0, 2, 9]),
    (10, [0, 3, 7, 9]),
    (10, [0, 4, 7, 9]),
    (10, [0, 5, 7, 9]),
    (10, [0, 6, 7, 9]),
    (10, [0, 1, 3, 9]),
    (10, [0, 2, 3, 9]),
    (10, [0, 1, 6, 9]),
    (10, [0, 2, 6, 9]),
    (11, [0, 2]),
    (11, [0, 7]),
    (11, [0, 4, 9]),
    (11, [0, 5, 9]),
    (11, [0, 6, 9]),
    (11, [0, 7, 9]),
    (11, [0, 8, 9]),
    (11, [0, 1, 10]),
    (11, [0, 2, 10]),
    (11, [0, 3, 10]),
    (11, [0, 4, 10]),
    (11, [0, 5, 10]),
    (11, [0, 6, 10]),
    (11, [0, 8, 10]),
    (11, [0, 9, 10]),
    (11, [0, 1, 4, 9]),
    (11, [0, 2, 4, 9]),
    (11, [0, 3, 4, 9]),
    (11, [0, 1, 5, 9]),
    (11, [0, 2, 5, 9]),
    (11, [0, 3, 5, 9]),
    (11, [0, 4, 5, 9]),
    (11, [0, 6, 7, 9]),
    (11, [0, 1, 6, 9]),
    (11, [0, 2, 6, 9]),
    (11, [0, 3, 6, 9]),
    (11, [0, 4, 6, 9]),
    (11, [0, 5, 6, 9]),
    (11, [0, 7, 8, 9]),
    (11, [0, 1, 7, 9]),
    (11, [0, 2, 7, 9]),
    (11, [0, 3, 7, 9]),
    (11, [0, 4, 7, 9]),
    (11, [0, 5, 7, 9]),
    (11, [0, 6, 7, 9]),
]

def _generate_mls(taps, m):
    """Generate a maximal length sequence (MLS) of length 2^m - 1."""
    period = 2**m - 1
    # Initialize state with 1 followed by zeros
    state = np.zeros(m, dtype=np.int8)
    state[0] = 1
    
    seq = np.zeros(period, dtype=np.int8)
    
    for i in range(period):
        seq[i] = state[0]
        # Compute new bit
        # XOR of bits at tap positions
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        
        # Shift left
        state[1:] = state[:-1]
        state[0] = new_bit
        
    return seq

def construct():
    best_score = float('inf')
    best_heights = None
    
    for m, taps in PRIMITIVE_POLYS:
        period = 2**m - 1
        if period < N:
            continue
            
        mls = _generate_mls(taps, m)
        # Map 0/1 to -1/1
        # mls is int8 0/1
        mls_pm1 = 2.0 * mls.astype(np.float64) - 1.0
        
        # Iterate over all possible starting phases
        for shift in range(period):
            # Extract window of length N
            # Use circular indexing
            idx = (np.arange(N) + shift) % period
            h = mls_pm1[idx]
            
            # Compute aperiodic autocorrelation
            # R[k] = sum h[i] h[i+k]
            ac = np.correlate(h, h, mode='full')
            
            # The peak is at index N-1
            # We want max abs of all other lags
            # Exclude the center
            ac_non_peak = np.concatenate([ac[:N-1], ac[N:]])
            
            max_ac = np.max(np.abs(ac_non_peak))
            
            # Score is the upper bound on C3, which is max_ac / N
            score = max_ac / N
            
            if score < best_score:
                best_score = score
                best_heights = h.copy()
                
    if best_heights is None:
        # Fallback to random if no valid MLS found (should not happen)
        best_heights = np.random.choice([-1.0, 1.0], size=N)
        
    return best_heights

# EVOLVE-BLOCK-END
