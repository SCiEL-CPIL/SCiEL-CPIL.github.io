"""Model-and-optimize: parameterize s and fit roots directly by precomputing square roots once."""
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    p = 1009
    # Precompute a root for every nonzero quadratic residue mod p.
    # root_of_res[r] = x such that x^2 % p == r, with 1 <= x <= p-1.
    # We pick the smallest such root for determinism and compactness.
    root_of_res = [0] * (p + 1)
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if root_of_res[r] == 0:  # keep the first (smallest x) root seen
            root_of_res[r] = x
    
    # Now scan s from 1 upward; the first s where s, s+1, ..., s+4 are all
    # quadratic residues (nonzero) is the minimal valid start.
    # Since p=1009 is prime, there are (p-1)/2 = 504 nonzero quadratic residues.
    # We need 5 consecutive integers all in that set.
    
    # Build a boolean membership array for speed: is_res[r] = True if r is a nonzero QR.
    is_res = [False] * (p + 5)  # extra slots to avoid bounds check
    for r in range(1, p):
        if root_of_res[r] != 0:
            is_res[r] = True
    
    # Scan for minimal s
    for s in range(1, p - 4 + 1):  # s from 1 to 1004
        # Check s, s+1, s+2, s+3, s+4
        if (is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]):
            result_roots = [root_of_res[s+i] for i in range(5)]
            return {'start': s, 'roots': result_roots}
    
    # Fallback (should never happen for p=1009)
    raise RuntimeError('No run of 5 consecutive nonzero quadratic residues found')
