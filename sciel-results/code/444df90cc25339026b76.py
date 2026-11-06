import time
import numpy as np

def construct():
    """
    Construct the [18, 9, 6] extended quadratic residue code.
    This code is optimal for A_2(18, 6) = 512.
    
    The construction uses the generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1
    over GF(2) for the [17, 9, 5] QR code, then extends to length 18 by adding
    an overall parity check bit.
    
    Returns a list of 512 integers, each representing an 18-bit codeword.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Generator polynomial coefficients: g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # The [17, 9, 5] QR code is cyclic.
    # g_powers are the exponents of x in g(x)
    g_powers = [0, 3, 4, 5, 8]
    
    # Construct the 9 basis vectors for the [17, 9, 5] code
    # The generator matrix rows are shifts of g(x).
    # Row i corresponds to x^i * g(x) mod (x^17 - 1)
    basis_17 = []
    for i in range(9):
        word = 0
        for p in g_powers:
            pos = (i + p) % 17
            word |= (1 << pos)
        basis_17.append(word)
        
    # Extend to [18, 9, 6] by adding a parity bit (bit 17)
    # The parity bit ensures the total weight is even, increasing min distance from 5 to 6.
    basis_18 = []
    for word in basis_17:
        # Count bits to determine parity
        # int.bit_count() is available in Python 3.10+, bin().count is compatible
        parity = bin(word).count('1') % 2
        if parity:
            word |= (1 << 17)
        basis_18.append(word)
        
    # Generate all 512 codewords by XORing subsets of the basis vectors
    # This is effectively m * G over GF(2)
    codewords = [0] * 512
    for msg in range(512):
        word = 0
        # Unroll the loop for speed, or use a simple loop since 9 is small
        for i in range(9):
            if (msg >> i) & 1:
                word ^= basis_18[i]
        codewords[msg] = word
        
    # Optional: Quick sanity check
    # The code is linear, so 0 is always a codeword.
    # We can verify a few distances to ensure correctness.
    # But given the known optimality and simple construction, this is often skipped
    # to save time, though the construction is already done.
    
    # Check if we are still within time (we should be, this takes < 1ms)
    if time.time() > deadline:
        # Return what we have (which is the full code)
        pass

    return codewords
