"""Find the minimum start s for 5 consecutive quadratic residues mod 1009 using bitset AND."""
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s wall-clock kill
    p = 1009
    m = 5

    # Build a bitset: bit v is set iff v is a non-zero quadratic residue mod p
    # We use a Python int as a big bitset
    qr_bits = 0
    roots = [0] * p  # roots[v] = one x such that x^2 % p == v

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # exclude zero residue
            qr_bits |= (1 << r)
            if roots[r] == 0:
                roots[r] = x

    # For each offset i in 0..4, we need bit s to be set iff (s+i) is a QR
    # That is: shifted_bits_i = qr_bits >> i  (bit s of shifted = bit (s+i) of qr_bits)
    # Then AND all shifted versions together
    combined = qr_bits
    for i in range(1, m):
        combined &= (qr_bits >> i)

    # Find the lowest set bit in combined, but only consider positions 1..1004
    # (s must be >= 1 and s+4 <= 1008, so s <= 1004)
    # Mask to only positions 1..1004
    mask = 0
    for s in range(1, p - m + 1):  # s from 1 to 1004
        mask |= (1 << s)
    combined &= mask

    if combined == 0:
        raise RuntimeError('No run of 5 consecutive non-zero QRs found')

    # Find the lowest set bit
    # In Python: (combined & -combined).bit_length() - 1 gives the index of the lowest set bit
    lowest_bit = (combined & -combined).bit_length() - 1
    best_s = lowest_bit

    best_roots = [roots[best_s + i] for i in range(m)]

    return {'start': best_s, 'roots': best_roots}
