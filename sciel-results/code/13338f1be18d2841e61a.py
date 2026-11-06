"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    # Precompute quadratic residues and one root for each non-zero residue
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # Ascending search for the minimal start s
    # s must be in [1, 1004] so that s+4 <= 1008 (since residues are 1..1008)
    for s in range(1, 1005):
        if all((s + i) in roots for i in range(5)):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
    
    # Fallback (should not happen for p=1009)
    raise RuntimeError("No valid run found")
# EVOLVE-BLOCK-END
