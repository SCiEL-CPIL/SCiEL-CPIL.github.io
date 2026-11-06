"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Precompute quadratic residues as a bitset: bit i is set if i is a non-zero QR mod p
    # Also build a root lookup table: root[r] = x where x^2 mod p == r (smallest such x)
    root = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if root[r] == 0:
            root[r] = x
    
    # Build bitset: bit i set iff i is a non-zero quadratic residue
    bitset = 0
    for x in range(1, p):
        r = (x * x) % p
        bitset |= (1 << r)
    
    # Find the smallest s such that s, s+1, ..., s+4 are all QR
    # s+4 <= 1008 => s <= 1004, and s >= 1
    # Check: (bitset >> s) & 0b11111 == 0b11111
    mask = 0b11111
    for s in range(1, 1005):
        if time.time() > deadline:
            raise RuntimeError('Time limit exceeded')
        if (bitset >> s) & mask == mask:
            # Find roots for each residue using the precomputed root table
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
