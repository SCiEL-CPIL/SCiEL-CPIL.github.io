import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0
    p = 1009
    m = 5

    # Precompute roots: smallest x in [1, p-1] such that x^2 mod p == r
    # We iterate x from 1 to p-1, so the first x found for a residue r is the smallest.
    root_map = {}
    for x in range(1, p):
        if time.time() > deadline:
            # This should practically never happen for p=1009, but safe to include
            break
        r = (x * x) % p
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x

    # Set for O(1) residue membership check
    qr_set = set(root_map.keys())

    # Search for the minimum s in [1, p-m] such that s..s+m-1 are all nonzero QRs
    # s must be <= p-m to ensure s+m-1 <= p-1 (1008), avoiding wraparound to 0 or negative
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        
        # Check if s, s+1, ..., s+m-1 are all in qr_set
        valid = True
        for i in range(m):
            if (s + i) not in qr_set:
                valid = False
                break
        
        if valid:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback if no solution found (should not happen for p=1009, m=5)
    raise RuntimeError("No run of 5 consecutive nonzero quadratic residues found")

# EVOLVE-BLOCK-END
