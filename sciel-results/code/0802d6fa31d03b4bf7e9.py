import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Fast popcount for 18-bit values using 3x6-bit lookup
    pop6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)

    def popcount18(arr):
        lo = (arr & 0x3F).astype(np.intp)
        mid = ((arr >> 6) & 0x3F).astype(np.intp)
        hi = ((arr >> 12) & 0x3F).astype(np.intp)
        return pop6[lo].astype(np.uint8) + pop6[mid].astype(np.uint8) + pop6[hi].astype(np.uint8)

    # ============================================================
    # Construction 1: Extended Quadratic Residue Code [18,9,6]
    # This is the known optimal construction for A_2(18,6) = 512
    # ============================================================
    #
    # The QR code of length 17 over GF(2) is a [17,9,5] code.
    # Its generator polynomial is g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # (one of the two irreducible degree-8 factors of x^17 - 1 over GF(2)).
    #
    # We generate all 2^9 = 512 codewords of the [17,9,5] code,
    # then extend each by appending a parity bit to get [18,9,6].
    #
    # The generator polynomial g(x) has degree 8, so the code has dimension 17-8=9.
    # We represent codewords as c(x) = m(x) * g(x) mod (x^17 - 1), where deg(m) < 9.
    #
    # Since x^17 ≡ 1 (mod x^17-1), we can compute m(x)*g(x) mod (x^17-1) by
    # reducing powers x^k for k >= 17 to x^(k-17).

    # Generator polynomial coefficients (bit i is coefficient of x^i)
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Bits: position 0,1,3,4,5,7,8 are set
    g_poly = 0
    for bit_pos in [0, 1, 3, 4, 5, 7, 8]:
        g_poly |= (1 << bit_pos)

    # Multiply m(x) * g(x) mod (x^17 - 1)
    # m(x) has degree < 9, g(x) has degree 8, so product has degree < 17.
    # No reduction needed! The product naturally fits in 17 bits.
    #
    # So for each message m (9-bit value), codeword = m * g_poly (polynomial mult mod x^17-1)
    # But since deg(m*g) < 17, it's just polynomial multiplication without reduction.

    def poly_mul_17(m, g):
        """Multiply two polynomials over GF(2), result mod (x^17 - 1).
        m is 9-bit, g is 9-bit (degree <= 8), result is 17-bit."""
        result = 0
        for i in range(9):  # bits of m
            if m & (1 << i):
                result ^= (g << i)
        # Reduce mod x^17 - 1: bits >= 17 wrap around
        # But since deg(m) < 9 and deg(g) < 9, deg(product) < 17, so no reduction needed.
        return result & 0xFFFF  # mask to 16 bits? No, need 17 bits.

    # Actually let me be more careful. Let me just do it properly.
    # m is 9 bits (positions 0..8), g is 9 bits (positions 0..8)
    # Product has bits up to position 16, so 17 bits total.

    def poly_mul_17(m, g):
        result = 0
        for i in range(9):
            if m & (1 << i):
                result ^= (g << i)
        return result & 0x1FFFF  # mask to 17 bits

    # Generate all 512 codewords of the [17,9,5] QR code
    qr_codewords_17 = []
    for m in range(512):
        cw = poly_mul_17(m, g_poly)
        qr_codewords_17.append(cw)

    # Extend to length 18 by appending parity bit
    # The parity bit makes the total weight even (since the [17,9,5] code has odd-weight codewords
    # that need extension to get distance 6).
    # Actually, for the extended QR code, we append a bit such that the total weight is even.
    # This is the standard extension: append bit = weight(cw) mod 2.

    qr_codewords_18 = []
    for cw in qr_codewords_17:
        parity = cw.bit_count() % 2
        extended = (cw << 1) | parity  # shift left and append parity bit
        # Wait, this puts the 17-bit codeword in the upper bits and parity in bit 0.
        # Let me use: extended = cw | (parity << 17)
        # Actually, let's keep it simple: the 18-bit word is cw (lower 17 bits) + parity (bit 17)
        extended = cw | (parity << 17)
        qr_codewords_18.append(extended)

    # Verify: check minimum distance
    # For a linear code, min distance = min weight of nonzero codewords
    min_dist_found = N + 1
    for cw in qr_codewords_18:
        if cw != 0:
            w = cw.bit_count()
            if w < min_dist_found:
                min_dist_found = w

    # Check if our construction gives distance >= 6
    if min_dist_found >= MIN_DIST:
        # Verify all pairs (for a linear code, min distance = min nonzero weight, but let's be safe)
        # For 512 words, checking all pairs is 512*511/2 = ~130k pairs, fast enough
        qr_valid = True
        words_arr = np.array(qr_codewords_18, dtype=np.uint32)
        for i in range(len(words_arr)):
            if time.time() > deadline - 10.0:
                break
            xor_vals = np.bitwise_xor(words_arr, np.uint32(words_arr[i]))
            bit_counts = popcount18(xor_vals)
            if bit_counts.min() < MIN_DIST:
                qr_valid = False
                break

        if qr_valid:
            return qr_codewords_18

    # If the QR construction didn't work as expected, fall back to greedy search
    # Sort candidates by weight (excluding 0)
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (x.bit_count(), x))

    best_words = [0]
    best_len = 1

    rng = np.random.default_rng(42)

    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break

        if attempt == 0:
            candidates = all_candidates
        else:
            candidates = all_candidates[:]
            i = 0
            while i < len(candidates):
                j = i + 1
                while j < len(candidates) and candidates[j].bit_count() == candidates[i].bit_count():
                    j += 1
                rng.shuffle(candidates[i:j])
                i = j

        words_list = [0]

        for c in candidates:
            if time.time() > deadline - 5.0:
                break

            words_arr = np.array(words_list, dtype=np.uint32)
            xor_vals = np.bitwise_xor(words_arr, np.uint32(c))
            bit_counts = popcount18(xor_vals)

            if bit_counts.min() >= MIN_DIST:
                words_list.append(c)

        if len(words_list) > best_len:
            best_len = len(words_list)
            best_words = words_list[:]

        if best_len >= 500:
            break

    return best_words
