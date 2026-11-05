"""Construct the [18, 9, 6] extended quadratic-residue code (A_2(18,6) = 512)."""
# EVOLVE-BLOCK-START
import time


def construct():
    # Deadline guard: 85 seconds to leave margin for evaluation overhead
    start_time = time.time()
    deadline = start_time + 85.0

    # Construct the [18, 9, 6] extended quadratic residue code.
    # This code has 512 codewords and minimum distance 6.

    # Generator polynomial for the [17, 9, 5] QR code:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients for x^0 to x^16
    g_coeffs = [0] * 17
    g_coeffs[0] = 1   # x^0
    g_coeffs[3] = 1   # x^3
    g_coeffs[4] = 1   # x^4
    g_coeffs[5] = 1   # x^5
    g_coeffs[8] = 1   # x^8

    # Precompute the 9 generator rows (cyclic shifts of g) for fast encoding
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
        if time.time() > deadline:
            break
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
