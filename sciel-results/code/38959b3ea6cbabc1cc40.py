"""
Construct the [18, 9, 6] extended quadratic-residue code (A_2(18,6) = 512).

Strategy:
  1) Directly construct the [18,9,6] extended QR code via the generator polynomial
     of the [17,9,5] QR code over GF(2), extended by an overall parity bit.
  2) This is a known optimal construction: 512 codewords, minimum distance 6.
  3) Verify all pairs have Hamming distance >= 6 using integer XOR and bit_count.
  4) Return the list of 512 codewords.

The construction uses the generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1
for the [17,9,5] QR code. Each message of length 9 is encoded by circular
convolution with g(x) mod (x^17 - 1), then a parity bit is appended.
"""

import time

N = 18
MIN_D = 6


def _construct_extended_qr18():
    """
    Construct the [18,9,6] extended quadratic-residue code.
    
    Uses the generator polynomial for the [17,9,5] QR code:
    g(x) = x^8 + x^5 + x^4 + x^3 + 1
    
    Steps:
    1. For each of the 2^9 = 512 messages, perform circular convolution
       with g(x) modulo (x^17 - 1) to get a 17-bit codeword.
    2. Append an overall parity bit to extend to 18 bits.
    3. Return all 512 codewords as integers.
    """
    # Generator polynomial coefficients: g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients for x^0, x^1, ..., x^16
    g_coeffs = [0] * 17
    g_coeffs[0] = 1   # x^0
    g_coeffs[3] = 1   # x^3
    g_coeffs[4] = 1   # x^4
    g_coeffs[5] = 1   # x^5
    g_coeffs[8] = 1   # x^8
    
    # Precompute the generator rows for each basis vector (shift of g)
    # row_i is the codeword corresponding to message = 2^i
    # This is the i-th cyclic shift of g
    gen_rows = []
    for i in range(9):
        row = [0] * 17
        for j in range(17):
            if g_coeffs[j]:
                row[(i + j) % 17] = 1
        # Convert to integer
        word = 0
        for pos, bit in enumerate(row):
            if bit:
                word |= (1 << pos)
        gen_rows.append(word)
    
    # Generate all 512 codewords by XOR combinations of the 9 generator rows
    codewords = []
    for msg in range(512):
        word = 0
        for i in range(9):
            if (msg >> i) & 1:
                word ^= gen_rows[i]
        
        # Compute parity bit for extension to length 18
        parity = word.bit_count() % 2
        if parity:
            word |= (1 << 17)
        
        codewords.append(word)
    
    return codewords


def _verify_code(words):
    """Verify all pairs have Hamming distance >= MIN_D. Return True if valid."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if (words[i] ^ words[j]).bit_count() < MIN_D:
                return False
    return True


def construct():
    """
    Construct the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6.
    
    Returns:
        A nonempty list of distinct integers in [0, 2**18). Each integer encodes
        one 18-bit binary codeword. Every pair has Hamming distance >= 6.
    
    Score: number of codewords (512 for the optimal [18,9,6] QR code).
    """
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s limit
    
    # Construct the optimal [18,9,6] extended QR code
    code = _construct_extended_qr18()
    
    # Verify validity
    if _verify_code(code):
        return code
    
    # Fallback (should not happen for the QR code)
    # Return a trivial valid code
    return [0]
