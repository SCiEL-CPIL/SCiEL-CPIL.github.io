"""
Program: Deterministic LFSR-based m-sequence initialization with targeted local search
to minimize the L2 norm of autocorrelation sidelobes for a signed step function of
length 400 on [-1/4, 1/4].

The metric chosen is the L2 norm of the off-center autocorrelation sidelobes:
    bound = sqrt(sum_{k=1}^{N-1} acf[k]^2)
where acf is the linear autocorrelation of the sequence.

Strategy:
1. Generate candidate m-sequences from all primitive polynomials of degree 8
   (m_len=255) and degree 9 (m_len=511). Since N=400, we take windows of length 400
   from these sequences (for degree 8, we concatenate two periods and take a window;
   for degree 9, we take a window directly).
2. Evaluate the bound for each candidate window (try multiple offsets/phases).
3. Select the best candidate as the starting point.
4. Perform a limited local search (bit flipping) to further reduce the bound.
5. Return the best sequence found.

This approach is deterministic, fast, and leverages the known near-optimality of
m-sequences for periodic autocorrelation. The linear autocorrelation of a window
of an m-sequence is also well-behaved.
"""

import numpy as np
import time
import itertools

N = 400

def compute_autocorrelation_bound(f):
    """
    Compute the L2 norm of the autocorrelation sidelobes.
    f: array of length N with values in {-1, 1}
    Returns: sqrt(sum_{k=1}^{N-1} acf[k]^2)
    """
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    bound = np.sqrt(np.sum(acf[1:] ** 2))
    return bound

def generate_lfsr_sequence(degree, poly, init_state):
    """
    Generate an m-sequence of length 2^degree - 1 using a linear feedback shift register.
    
    Args:
        degree: degree of the primitive polynomial
        poly: integer representing the polynomial (bit i set means x^i is present)
              The polynomial should be primitive and of the given degree.
              The feedback taps are determined by the polynomial.
        init_state: initial state of the LFSR (non-zero integer < 2^degree)
    
    Returns:
        array of length 2^degree - 1 with values in {-1, 1}
    """
    m_len = (1 << degree) - 1
    seq = np.zeros(m_len, dtype=np.int8)
    state = init_state
    
    # Extract feedback taps from polynomial
    # The polynomial is of the form x^degree + c_{degree-1} x^{degree-1} + ... + c_0
    # For the LFSR, the feedback is computed from the taps.
    # We use the standard Fibonacci LFSR.
    # The polynomial bits (excluding the leading x^degree term) give the taps.
    # Let's define taps as the positions where the coefficient is 1.
    # For a polynomial like x^8 + x^4 + x^3 + x^2 + 1, the taps are [0, 2, 3, 4].
    # The feedback bit is XOR of bits at positions corresponding to taps (shifted).
    
    # Actually, let's use a simpler approach:
    # The polynomial is given as an integer where bit i represents x^i.
    # For degree d, the polynomial has bit d set (the leading term) and possibly others.
    # The feedback is computed from the lower d bits.
    # Standard Fibonacci LFSR:
    # state is d bits. At each step, output the LSB (or MSB, depending on convention).
    # new bit = XOR of bits at positions where the polynomial has 1s (excluding the leading term).
    
    # Let's use the convention where the state is a d-bit number.
    # The output is the LSB of the state.
    # The new bit is the XOR of the bits in the state at positions i where poly has bit i set,
    # for i < d. (The leading term x^d is implicit.)
    
    # Actually, the standard way:
    # Given polynomial p(x) = x^d + c_{d-1}x^{d-1} + ... + c_0
    # The LFSR recurrence is: s_{n+d} = c_{d-1} s_{n+d-1} + ... + c_0 s_n (mod 2)
    # In terms of the state register, if we store s_n, s_{n+1}, ..., s_{n+d-1},
    # then the new bit s_{n+d} is the XOR of s_{n+i} for i where c_i = 1.
    
    # Let's define the taps as the list of i in [0, d-1] where the polynomial has bit i set.
    taps = []
    for i in range(degree):
        if (poly >> i) & 1:
            taps.append(i)
    
    # Generate the sequence
    for i in range(m_len):
        # Output is the LSB
        seq[i] = 1 if (state & 1) else -1
        
        # Compute new bit
        new_bit = 0
        for tap in taps:
            new_bit ^= (state >> tap) & 1
        
        # Shift left and insert new bit at the MSB
        state = (state << 1) | new_bit
        state &= (1 << degree) - 1  # Keep only d bits
    
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Primitive polynomials for degrees 8 and 9
    # Degree 8 primitive polynomials (in integer form, bit i = x^i)
    # x^8 + x^4 + x^3 + x^2 + 1 = 0b100111001 = 313
    # x^8 + x^5 + x^3 + x^2 + 1 = 0b100101101 = 301
    # x^8 + x^6 + x^5 + x^4 + 1 = 0b111100001 = 481
    # x^8 + x^7 + x^5 + x^3 + 1 = 0b110101001 = 425
    # x^8 + x^7 + x^5 + x^2 + 1 = 0b110100101 = 421
    # x^8 + x^7 + x^3 + x^2 + 1 = 0b110011001 = 409
    # x^8 + x^6 + x^5 + x^3 + 1 = 0b111010001 = 465
    # x^8 + x^6 + x^3 + x^2 + 1 = 0b110011001 = 409 (duplicate)
    # Let's use a few known ones.
    
    poly8_candidates = [
        0b100111001,  # 313
        0b100101101,  # 301
        0b111100001,  # 481
        0b110101001,  # 425
        0b110100101,  # 421
        0b110011001,  # 409
        0b111010001,  # 465
        0b101101001,  # 369
        0b101011001,  # 345
        0b101001101,  # 333
    ]
    
    # Degree 9 primitive polynomials
    # x^9 + x^5 + x^3 + x^2 + 1 = 0b10010110101 = 1213
    # x^9 + x^7 + x^5 + x^4 + 1 = 0b11110000101 = 1925
    # x^9 + x^8 + x^6 + x^3 + 1 = 0b11010010001 = 1673
    # x^9 + x^8 + x^5 + x^3 + 1 = 0b11010100001 = 1697
    # x^9 + x^8 + x^4 + x^2 + 1 = 0b11001010001 = 1617
    # x^9 + x^7 + x^6 + x^4 + 1 = 0b11101000101 = 1861
    # x^9 + x^7 + x^4 + x^2 + 1 = 0b11010010001 = 1673 (duplicate)
    # Let's use a few.
    
    poly9_candidates = [
        0b10010110101,  # 1213
        0b11110000101,  # 1925
        0b11010010001,  # 1673
        0b11010100001,  # 1697
        0b11001010001,  # 1617
        0b11101000101,  # 1861
        0b10110100101,  # 1445
        0b10101100101,  # 1397
        0b10100110101,  # 1357
        0b10011010101,  # 1245
    ]
    
    best_f = None
    best_bound = float('inf')
    
    # Try degree 8 sequences
    m_len8 = 255
    for poly in poly8_candidates:
        # Try a few initial states
        for init_state in [1, 3, 7, 15, 31, 63, 127, 255]:
            seq = generate_lfsr_sequence(8, poly, init_state)
            # Create a longer sequence by repeating
            long_seq = np.tile(seq, 3)  # 765 elements
            # Try all windows of length N
            for offset in range(0, 300):  # Sample a subset of offsets
                window = long_seq[offset:offset+N]
                bound = compute_autocorrelation_bound(window)
                if bound < best_bound:
                    best_bound = bound
                    best_f = window.copy()
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Try degree 9 sequences
    if time.time() < deadline:
        m_len9 = 511
        for poly in poly9_candidates:
            for init_state in [1, 3, 7, 15, 31, 63, 127, 255]:
                seq = generate_lfsr_sequence(9, poly, init_state)
                # Try all windows of length N
                for offset in range(0, 112):  # 511 - 400 + 1 = 112 possible offsets
                    window = seq[offset:offset+N]
                    bound = compute_autocorrelation_bound(window)
                    if bound < best_bound:
                        best_bound = bound
                        best_f = window.copy()
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Fallback if no sequence was found (should not happen)
    if best_f is None:
        rng = np.random.default_rng(42)
        best_f = rng.choice([-1, 1], size=N)
        best_bound = compute_autocorrelation_bound(best_f)
    
    # Local search polish
    if time.time() < deadline:
        f = best_f.copy()
        bound = best_bound
        rng = np.random.default_rng(123)
        
        # Use a simple hill climbing with random restarts if stuck
        improved = True
        max_no_improve = 5000
        no_improve_count = 0
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            idx = rng.integers(0, N)
            old_val = f[idx]
            f[idx] = -f[idx]
            new_bound = compute_autocorrelation_bound(f)
            
            if new_bound < bound:
                bound = new_bound
                best_f = f.copy()
                best_bound = bound
                no_improve_count = 0
            else:
                f[idx] = old_val
                no_improve_count += 1
        
        # If we used up the no_improve budget, try a few more random restarts
        while time.time() < deadline:
            # Try flipping a random subset of bits
            num_flips = rng.integers(1, 10)
            indices = rng.choice(N, size=num_flips, replace=False)
            old_vals = f[indices].copy()
            f[indices] = -f[indices]
            new_bound = compute_autocorrelation_bound(f)
            
            if new_bound < best_bound:
                best_bound = new_bound
                best_f = f.copy()
            else:
                f[indices] = old_vals
    
    return best_f

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_autocorrelation_bound(f)}")
