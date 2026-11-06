"""Early runs of five quadratic residues modulo 1009."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009

    # Build a bitset of quadratic residues mod p (excluding 0).
    # Bit i is 1 iff i is a non-zero quadratic residue mod p.
    residue_bits = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_bits |= (1 << r)

    # A start s works iff bits s..s+4 are all set.
    # Compute the set of all such s in one shot via shifted ANDs.
    combined = (
        residue_bits
        & (residue_bits >> 1)
        & (residue_bits >> 2)
        & (residue_bits >> 3)
        & (residue_bits >> 4)
    )

    # Restrict to valid range 1 <= s <= 1004.
    # Build a mask with bits 1..1004 set.
    valid_mask = ((1 << 1005) - 1) ^ 1  # bits 1..1004 set (bit 0 cleared)

    candidates = combined & valid_mask
    if candidates == 0:
        raise RuntimeError('No valid run found')

    # Find the lowest set bit -> gives the minimal s.
    lowest = candidates & (-candidates)
    s = lowest.bit_length() - 1

    # Now find a square root for each of s, s+1, ..., s+4.
    # Build a map from residue to one of its roots (any root in 1..p-1).
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x

    root_list = [roots[s + i] for i in range(5)]

    return {'start': s, 'roots': root_list}
# EVOLVE-BLOCK-END
