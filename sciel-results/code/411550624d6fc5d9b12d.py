"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s wall-clock limit
    
    # Precompute quadratic residues and their roots
    # For each residue r (1..1008), store a root x such that x*x % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # We need to find the smallest s (1 <= s <= 1004) such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues mod p (non-zero)
    # i.e., all in roots (and none equal to 0)
    
    # Build a boolean array for quick lookup
    is_residue = [False] * (p + 5)  # extra space to avoid wraparound checks
    for r in roots:
        if r != 0:
            is_residue[r] = True
    
    # Search for the minimum s
    best_s = None
    best_roots = None
    
    for s in range(1, p - 4):  # s from 1 to 1004
        if time.time() > deadline:
            break
        if all(is_residue[s + i] for i in range(5)):
            best_s = s
            best_roots = [roots[s + i] for i in range(5)]
            break
    
    if best_s is None:
        # Fallback: use the original descending search if we didn't find one
        # (shouldn't happen, but for safety)
        for s in range(p - 5, 0, -1):
            if time.time() > deadline:
                break
            if all(is_residue[s + i] for i in range(5)):
                best_s = s
                best_roots = [roots[s + i] for i in range(5)]
                break
    
    if best_s is None:
        raise RuntimeError('No run found')
    
    return {'start': best_s, 'roots': best_roots}
# EVOLVE-BLOCK-END
