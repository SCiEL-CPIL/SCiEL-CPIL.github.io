"""Find minimum s such that s, s+1, s+2, s+3, s+4 are all
non-zero quadratic residues mod 1009, each certified by a root in [1,1008]."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute: for each non-zero residue r in [1, p-1], store a root x in [1, p-1]
    # such that x^2 % p == r. Keep the smallest x for determinism.
    root_of = {}
    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r != 0 and r not in root_of:
            root_of[r] = x

    # Build a bitset of quadratic residues using Python's arbitrary-precision integers.
    # bit i is 1 iff i is a non-zero quadratic residue mod p.
    bitset = 0
    for r in root_of:
        bitset |= (1 << r)

    # We need to find the smallest s in [1, p-m] such that bits s through s+m-1 are all set.
    # Create a mask with m consecutive 1s: mask = (1 << m) - 1
    mask = (1 << m) - 1  # 0b11111 for m=5

    # For each candidate s, check if (bitset >> s) & mask == mask
    # This means bits s, s+1, ..., s+m-1 are all set.
    # We scan s from 1 upward to find the MINIMUM s.
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        # Check if bits s through s+m-1 are all set
        if ((bitset >> s) & mask) == mask:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback: should not happen, but return a valid late run if we must.
    for s in range(p - m, 0, -1):
        if time.time() > deadline:
            break
        ok = True
        for i in range(m):
            val = s + i
            if val not in root_of:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(m)]}

    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
