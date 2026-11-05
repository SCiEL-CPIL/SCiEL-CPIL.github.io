"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Construct the [18, 9, 6] extended quadratic residue code.
    # This code has 512 codewords and minimum distance 6.
    # Generator polynomial for the [17, 9, 5] QR code:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    
    # Precompute the 9 generator rows (cyclic shifts of g)
    g_coeffs = [0] * 17
    g_coeffs[0] = 1
    g_coeffs[3] = 1
    g_coeffs[4] = 1
    g_coeffs[5] = 1
    g_coeffs[8] = 1
    
    gen_rows = []
    for i in range(9):
        word = 0
        for j in range(17):
            if g_coeffs[j]:
                word |= (1 << ((i + j) % 17))
        gen_rows.append(word)
    
    # Generate all 512 codewords by XOR combinations of the 9 generator rows,
    # then extend with an overall parity bit to length 18.
    words = []
    for msg in range(512):
        word = 0
        for i in range(9):
            if (msg >> i) & 1:
                word ^= gen_rows[i]
        parity = word.bit_count() % 2
        if parity:
            word |= (1 << 17)
        words.append(word)
    
    # Verify validity: all pairs must have Hamming distance >= 6
    n = len(words)
    valid = True
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if (words[i] ^ words[j]).bit_count() < 6:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return words
    
    # Fallback (should not happen for the QR code): return a trivial valid code
    return [0]
# EVOLVE-BLOCK-END
