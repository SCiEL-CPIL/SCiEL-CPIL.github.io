"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85  # safety margin before 90s wall-clock
    
    # Precompute: for each residue r in 1..1008, store one root x in 1..1008
    # such that x*x % p == r. Non-residues get None.
    root_of = [None] * (p + 1)
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] is None:
            root_of[r] = x
    
    # Scan s from 1 to 1004, find the smallest s where s..s+4 are all non-zero residues
    for s in range(1, p - 4):
        if time.time() > deadline:
            # Fallback: return any valid run (shouldn't happen for this small problem)
            # But just in case, try a quick descending search
            for s2 in range(p - 5, 0, -1):
                if all(root_of[s2 + i] is not None for i in range(5)):
                    return {'start': s2, 'roots': [root_of[s2 + i] for i in range(5)]}
            raise RuntimeError('No run found before deadline')
        
        if (root_of[s] is not None and 
            root_of[s+1] is not None and 
            root_of[s+2] is not None and 
            root_of[s+3] is not None and 
            root_of[s+4] is not None):
            return {'start': s, 'roots': [root_of[s+i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
