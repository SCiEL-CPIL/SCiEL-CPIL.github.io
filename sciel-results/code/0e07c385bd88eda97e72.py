"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of non-zero quadratic residues modulo p.
    # bit i is 1 iff i is a non-zero QR mod p.
    # We only care about residues 1..p-1 (i.e., 1..1008).
    bitset = 0
    # Also build roots_lookup in the same pass to avoid redundant work
    roots_lookup = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            bitset |= (1 << r)
            if r not in roots_lookup:
                roots_lookup[r] = x
    
    # Compute consec: bit s is 1 iff bits s..s+4 are all set in bitset
    consec = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Find the smallest s in [1, 1004] such that bit s of consec is 1.
    # Use bit_length trick to find lowest set bit efficiently
    for s in range(1, 1005):
        if (consec >> s) & 1:
            roots = [roots_lookup[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of five consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
