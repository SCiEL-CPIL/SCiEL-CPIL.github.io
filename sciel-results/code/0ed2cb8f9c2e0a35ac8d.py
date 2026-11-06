"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    p = 1009
    m = 5
    
    # Precompute quadratic residues: for each residue value v in 1..1008,
    # store a representative root r such that r*r % p == v.
    # We want the smallest positive root for each residue to be deterministic.
    root_of = [0] * p  # index by residue value
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if root_of[r] == 0 or x < root_of[r]:
            root_of[r] = x
    
    # Search for minimum s in 1..1004 such that s, s+1, ..., s+4 are all QR
    # (and all within 1..1008, which is automatic since s+4 <= 1008 when s <= 1004)
    # Use a sliding window check for efficiency
    for s in range(1, 1005):
        ok = True
        roots = []
        for i in range(m):
            v = s + i
            if root_of[v] == 0:
                ok = False
                break
            roots.append(root_of[v])
        if ok:
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
