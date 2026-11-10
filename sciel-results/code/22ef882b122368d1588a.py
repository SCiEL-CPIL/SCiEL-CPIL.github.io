import time

def construct():
    p = 1009
    deadline = time.monotonic() + 85  # safety margin before 90s kill

    # Precompute a root for each non-zero quadratic residue modulo p
    # roots[res] = smallest positive r in 1..p-1 such that r*r % p == res, or 0 if not a QR
    roots = [0] * p
    for r in range(1, p):
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r

    # Build a bitset: bit i is set iff i is a non-zero QR mod p
    qr_bits = 0
    for i in range(1, p):
        if roots[i] != 0:
            qr_bits |= (1 << i)

    # Find positions s where bits s, s+1, s+2, s+3, s+4 are all set
    candidate_bits = qr_bits & (qr_bits >> 1) & (qr_bits >> 2) & (qr_bits >> 3) & (qr_bits >> 4)

    if candidate_bits != 0:
        # Find the lowest set bit
        lowest_bit = candidate_bits & -candidate_bits
        s = lowest_bit.bit_length() - 1

        # Validate s is in range 1..1004
        if 1 <= s <= 1004:
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }

    # Fallback: linear scan (should not be needed but ensures correctness)
    for s in range(1, 1005):
        if time.monotonic() > deadline:
            break
        if (roots[s] != 0 and roots[s+1] != 0 and roots[s+2] != 0 and
            roots[s+3] != 0 and roots[s+4] != 0):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }

    raise RuntimeError("No solution found")
