"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Precompute roots for each quadratic residue
    # roots[r] = x such that x*x % p == r, with 1 <= x <= p-1
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # Build a bitmask: bit i is set if residue (i+1) is a quadratic residue
    # We need 5 consecutive residues s, s+1, s+2, s+3, s+4 all to be residues
    # Using bitmask for fast checking
    
    # Create a boolean array: is_residue[i] for i in 0..p-1
    is_residue = [False] * p
    for r in roots:
        is_residue[r] = True
    
    # For each s from 1 to 1004, check if s, s+1, s+2, s+3, s+4 are all residues
    # Note: we need s+i <= 1008 (since ri must be 1..1008)
    # So s can be at most 1004 (s+4 = 1008)
    
    best_s = None
    best_roots = None
    
    for s in range(1, 1005):
        # Check if all 5 consecutive values are quadratic residues
        # s, s+1, s+2, s+3, s+4 must all be in 1..1008 and be residues
        if s + 4 > 1008:
            break
        
        if all(is_residue[s + i] for i in range(5)):
            best_s = s
            best_roots = [roots[s + i] for i in range(5)]
            break  # Since we're scanning in ascending order, first found is minimal
    
    if best_s is None:
        raise RuntimeError('No valid run found')
    
    return {'start': best_s, 'roots': best_roots}
# EVOLVE-BLOCK-END
