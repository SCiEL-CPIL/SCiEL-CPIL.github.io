"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Precompute: for each value v in 1..p-1, store a root r such that r^2 % p == v.
    # We want the smallest s (1 <= s <= 1004) such that s, s+1, ..., s+4 are all
    # non-zero quadratic residues mod p.
    
    # Build the set of quadratic residues (excluding 0)
    qr_set = set()
    roots = {}
    for x in range(1, p):
        r2 = (x * x) % p
        if r2 not in roots:
            roots[r2] = x
        qr_set.add(r2)
    
    # Now search for the minimum s such that s, s+1, ..., s+4 are all in qr_set
    # and none of them are 0 (which they can't be since s >= 1 and s+4 <= 1008)
    
    # Direct search from s=1 upward
    for s in range(1, p - m + 1):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check if s, s+1, ..., s+4 are all quadratic residues
        if all((s + i) in qr_set for i in range(m)):
            result_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': result_roots}
    
    # If we didn't find one (shouldn't happen), raise
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
