"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Precompute roots for each quadratic residue
    # roots[r] = x such that x*x % p == r, with 1 <= x <= p-1
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # Create a boolean array: is_residue[i] for i in 0..p-1
    is_residue = [False] * p
    for r in roots:
        is_residue[r] = True
    
    # Find the minimal s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # s must be in [1, 1004] to keep s+4 <= 1008
    for s in range(1, 1005):
        if all(is_residue[s + i] for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
