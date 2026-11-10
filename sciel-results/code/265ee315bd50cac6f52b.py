import time

def construct():
    p = 1009
    deadline = time.monotonic() + 85  # safety margin before 90s kill

    # Precompute a root for each quadratic residue modulo p
    # roots[res] = smallest positive r in 1..p-1 such that r*r % p == res, or 0
    roots = [0] * p
    for r in range(1, p):
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r

    # Create a bitset for QRs
    # bit i set to 1 if i is a non-zero QR
    qr_bits = 0
    for i in range(1, p):
        if roots[i] != 0:
            qr_bits |= (1 << i)

    # We need s, s+1, s+2, s+3, s+4 to be QRs.
    # This corresponds to bit s being set in the AND of qr_bits and its right shifts.
    candidate_bits = qr_bits & (qr_bits >> 1) & (qr_bits >> 2) & (qr_bits >> 3) & (qr_bits >> 4)
    
    if candidate_bits == 0:
        # Fallback to loop if bitset logic fails
        for s in range(1, 1005):
            if roots[s] and roots[s+1] and roots[s+2] and roots[s+3] and roots[s+4]:
                return {
                    'start': s,
                    'roots': [roots[s + i] for i in range(5)]
                }
        raise RuntimeError("No solution found")

    # Find the lowest set bit in candidate_bits
    # candidate_bits & -candidate_bits isolates the lowest set bit
    lowest_bit = candidate_bits & -candidate_bits
    s = lowest_bit.bit_length() - 1
    
    # Ensure s is in valid range 1..1004
    # Since bit 0 is not set in qr_bits (roots[0]=0), s will be >= 1.
    # If s > 1004, no valid solution in range (but problem guarantees one).
    if s > 1004:
        raise RuntimeError("No solution found in range")

    return {
        'start': s,
        'roots': [roots[s + i] for i in range(5)]
    }
