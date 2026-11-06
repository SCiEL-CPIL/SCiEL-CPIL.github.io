"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Step 1: Compute quadratic residues and store one root for each
    # For each x in 1..p-1, x^2 mod p is a quadratic residue
    # Store the smallest root for each residue
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue  # exclude zero residues
        if r not in roots:
            roots[r] = x
        elif x < roots[r]:
            roots[r] = x
    
    # Step 2: Scan from s=1 upward to find the first valid run of 5 consecutive residues
    # s must be in [1, 1004], and s+i must be in [1, 1008] for i=0..4
    # So s can be at most 1004 (since s+4 <= 1008)
    # We need s, s+1, s+2, s+3, s+4 all in roots (non-zero quadratic residues)
    
    # Optimization: use a boolean array for O(1) lookup
    is_residue = [False] * (p + 1)  # index 0..p
    for r in roots:
        is_residue[r] = True
    
    best_result = None
    for s in range(1, 1005):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
        if all(is_residue[s + i] for i in range(5)):
            best_result = {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
            break  # We want the minimal s, so stop at first found
    
    if best_result is None:
        # Fallback: descending search if ascending didn't find one (shouldn't happen)
        for s in range(1004, 0, -1):
            if time.time() > deadline:
                break
            if all(is_residue[s + i] for i in range(5)):
                best_result = {
                    'start': s,
                    'roots': [roots[s + i] for i in range(5)]
                }
                break
    
    if best_result is None:
        raise RuntimeError('No valid run found')
    
    return best_result
# EVOLVE-BLOCK-END
