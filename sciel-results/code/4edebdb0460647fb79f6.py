"""Find the minimum start s for 5 consecutive quadratic residues modulo 1009."""

import time

def construct():
    p = 1009
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Precompute: for each residue r in 1..p-1, store a root x such that x^2 % p == r
    # We only need non-zero residues (1 to p-1 = 1008)
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            roots[r] = x
    
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all in roots
    # and s+4 <= p-1 (i.e., s <= p-5 = 1004), with s >= 1
    # All values s..s+4 must be in 1..1008 (non-zero residues)
    
    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            # Should not happen for this small problem, but guard just in case
            raise RuntimeError("Deadline exceeded")
        
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues (non-zero)
        found = True
        for i in range(5):
            val = s + i
            if val < 1 or val > p - 1 or val not in roots:
                found = False
                break
        if found:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
