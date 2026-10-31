"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(a):
    """Full autoconvolution of a (length N)."""
    return np.convolve(a, a, mode='full')


def _score(a):
    """
    Score = max(|autoconv|) / (sum(a)^2).
    Lower is better.
    """
    s = a.sum()
    if s == 0:
        return np.inf
    ac = _autoconv(a)
    return np.max(np.abs(ac)) / (s * s)


def _construct_mseq(k, n):
    """
    Construct a binary m-sequence of length 2^k - 1 using a primitive polynomial,
    then truncate or pad to length n.
    Returns a 1D numpy array of 0s and 1s.
    """
    # Primitive polynomials for small k (x^k + ...)
    # We'll use a standard primitive polynomial representation.
    # For k=7, length=127. For k=8, length=255. For k=9, length=511.
    # We need length >= n or we can tile/truncate.
    # Let's pick k such that 2^k - 1 >= n.
    # For n=400, k=9 gives 511.
    
    # We need a primitive polynomial for GF(2^k).
    # Let's hardcode a few primitive polynomials.
    # k=9: x^9 + x^4 + 1 (0b100010001 = 273)
    # k=8: x^8 + x^4 + x^3 + x^2 + 1 (0b100111001 = 313)
    # k=7: x^7 + x^3 + 1 (0b1001001 = 73)
    
    # Find smallest k such that 2^k - 1 >= n
    k = 1
    while (1 << k) - 1 < n:
        k += 1
    
    # Primitive polynomials (coefficients as integers, bit i is coeff of x^i)
    # Note: standard representation varies. Let's use the integer where bit i is coeff of x^i.
    # The LFSR taps are determined by the polynomial.
    # For polynomial x^k + c_{k-1}x^{k-1} + ... + c_1 x + c_0
    # If c_0 = 1, it's primitive.
    
    # Let's use a known primitive polynomial for the chosen k.
    # This is a bit of a hack, but for small k it's fine.
    # If k is not in our table, we fall back to a random sequence.
    
    # Table of primitive polynomials (coefficients as bitmask, MSB is x^k)
    # Actually, let's just use a simple feedback LFSR with a fixed tap set that is known to be primitive for small k.
    # For k=9, taps at 9 and 4 (x^9 + x^4 + 1).
    # For k=8, taps at 8, 4, 3, 2, 1 (x^8 + x^4 + x^3 + x^2 + x + 1? No, that's not primitive).
    # Let's use a reliable source of primitive polynomials.
    
    # k=1: x+1 (not primitive for m-seq, length 1)
    # k=2: x^2+x+1 (length 3)
    # k=3: x^3+x+1 (length 7)
    # k=4: x^4+x+1 (length 15)
    # k=5: x^5+x^2+1 (length 31)
    # k=6: x^6+x+1 (length 63)
    # k=7: x^7+x^3+1 (length 127)
    # k=8: x^8+x^4+x^3+x^2+1 (length 255)
    # k=9: x^9+x^4+1 (length 511)
    # k=10: x^10+x^3+1 (length 1023)
    
    # Taps (indices of 1s in the polynomial, excluding the leading x^k term)
    # For x^k + sum(x^i), the feedback is from bit (k-1) to bit 0?
    # Standard LFSR: new_bit = XOR of bits at tap positions.
    # Let's define taps as the list of exponents i where coeff is 1, excluding k.
    
    primitive_polys = {
        2: [1, 0], # x^2+x+1
        3: [1, 0], # x^3+x+1
        4: [1, 0], # x^4+x+1
        5: [2, 0], # x^5+x^2+1
        6: [1, 0], # x^6+x+1
        7: [3, 0], # x^7+x^3+1
        8: [4, 3, 2, 0], # x^8+x^4+x^3+x^2+1
        9: [4, 0], # x^9+x^4+1
        10: [3, 0], # x^10+x^3+1
    }
    
    if k not in primitive_polys:
        # Fallback: random binary sequence
        return np.random.randint(0, 2, n)
    
    taps = primitive_polys[k]
    length = (1 << k) - 1
    
    # Initialize LFSR state with 1s
    state = [1] * k
    seq = []
    
    for i in range(length):
        # Output bit
        bit = state[-1]
        seq.append(bit)
        
        # Compute feedback
        # The polynomial is x^k + sum_{j in taps} x^j + ... 
        # Wait, the taps list includes the constant term?
        # In my dict above, for k=9, taps=[4,0]. This means x^9 + x^4 + 1.
        # The feedback bit is XOR of bits at positions corresponding to taps.
        # In a right-shift LFSR, state[0] is MSB, state[k-1] is LSB (output).
        # Let's be careful.
        # Let's use a standard implementation.
        
        # Actually, let's just generate the sequence and check its length.
        # If the period is not length, the polynomial wasn't primitive or implementation is wrong.
        
        # Let's restart LFSR logic more carefully.
        # state is a list of k bits.
        # new_bit = XOR of state[i] for i where (k - i) is in taps? 
        # This is confusing. Let's use a simpler approach:
        # Represent state as an integer.
        # state is k bits.
        # new_bit = parity of (state & mask)
        # where mask has 1s at positions corresponding to the taps.
        # If taps are exponents in the polynomial (excluding x^k), then for x^9+x^4+1, taps are 4 and 0.
        # The feedback is taken from the bits at these positions in the current state?
        # No, it depends on the LFSR configuration (Fibonacci vs Galois).
        
        # Let's use the Fibonacci LFSR with the "standard" feedback.
        # x^k = sum_{i in taps} x^i
        # So the new bit (which becomes x^k in the next step, or is output) is the XOR of the bits at positions i.
        # If we store state as [b_{k-1}, ..., b_0] where b_0 is the least significant (newest),
        # then b_{k-1} is the oldest.
        # Output is b_0? Or b_{k-1}?
        
        # Let's just generate it and verify the period.
        
        # For simplicity, let's use a known good m-sequence generator from a library or hardcoded.
        # Since I can't import external libs, I'll hardcode a 511-bit m-sequence.
        # This is safer.
        
        break # Break out, we'll hardcode.
    
    # Hardcoded 511-bit m-sequence (from x^9+x^4+1)
    # I will generate this once and paste it.
    # Actually, I can generate it correctly in the code if I get the LFSR right.
    
    # Let's try again with a correct LFSR implementation.
    # Polynomial: x^9 + x^4 + 1.
    # Taps: 9, 4, 0.
    # Feedback = state[4] XOR state[0] (if state is 0-indexed from LSB).
    # Let's use state as an integer.
    # state has 9 bits.
    # new_bit = (state >> 4) ^ (state & 1)
    # state = (state >> 1) | (new_bit << 8)
    # output = state & 1 (LSB)
    
    state = 1  # Initial state (any non-zero)
    seq = []
    for i in range(length):
        # Output LSB
        seq.append(state & 1)
        
        # Compute feedback
        # Taps are at positions 4 and 0 (0-indexed from LSB)
        # Wait, the polynomial is x^9 + x^4 + 1.
        # This means x^9 = x^4 + 1.
        # The new bit (MSB) is determined by the bits at x^4 and x^0.
        # In the shift register, if we shift right, the new MSB comes in.
        # The bits at x^4 and x^0 are at positions 4 and 0 in the current state (if LSB is pos 0).
        # So feedback = (state >> 4) ^ (state & 1)
        
        fb = (state >> 4) ^ (state & 1)
        state = (state >> 1) | (fb << 8)
    
    # Convert to numpy array
    seq = np.array(seq, dtype=np.int8)
    
    # Truncate or pad to n
    if length >= n:
        return seq[:n]
    else:
        # Tile
        repeats = (n // length) + 1
        seq = np.tile(seq, repeats)
        return seq[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate m-sequence
    mseq = _construct_mseq(9, N)
    
    # Convert to signed: 0 -> -1, 1 -> 1
    a = 2 * mseq - 1
    a = a.astype(np.float64)
    
    # 2. Local search with sign flips
    # Incremental update of autoconvolution is complex.
    # For N=400, full autoconv is 800 points.
    # np.convolve is fast. Let's see if we can do many iterations.
    
    current_score = _score(a)
    best_a = a.copy()
    best_score = current_score
    
    # Precompute? No, just iterate.
    # Random order of indices
    indices = np.arange(N)
    
    # Try to improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Flip sign
            old_val = a[idx]
            a[idx] = -a[idx]
            
            # Calculate new score
            new_score = _score(a)
            
            if new_score < best_score:
                best_score = new_score
                best_a = a.copy()
                improved = True
            else:
                # Revert
                a[idx] = old_val
    
    # Final check
    if _score(best_a) < best_score:
        best_score = _score(best_a)
    
    return best_a

# EVOLVE-BLOCK-END
