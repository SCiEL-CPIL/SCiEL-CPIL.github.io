"""Rewritten program to find the minimum run-start s for 5 consecutive quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Precompute a root for each non-zero quadratic residue modulo p.
    # We store the smallest x such that x^2 % p == r.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Store the first (smallest) root found for each residue
            if r not in roots:
                roots[r] = x
    
    # Search for the minimum s such that s, s+1, ..., s+m-1 are all non-zero quadratic residues.
    # s must be between 1 and p - m (since s+m-1 <= p-1 is not strictly required, 
    # but s+i must be in 1..p-1. Wait, the problem says 1<=s<=1004 and 1<=ri<=1008.
    # The residues are modulo p. So s+i can be any value 1..p-1.
    # If s+i == p, that's 0 mod p, which is excluded.
    # So s+i must be in {1, ..., p-1}.
    # Thus s+m-1 <= p-1 => s <= p-m.
    # p=1009, m=5, so s <= 1004. This matches the constraint 1<=s<=1004.
    
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+m-1 are all in roots
        valid = True
        for i in range(m):
            val = s + i
            if val not in roots:
                valid = False
                break
        
        if valid:
            res_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
    
    # Fallback, should not happen for this prime and m
    raise RuntimeError("No valid run found")
# EVOLVE-BLOCK-END
