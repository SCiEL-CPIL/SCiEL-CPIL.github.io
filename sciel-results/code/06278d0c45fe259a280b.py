"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute the set of nonzero quadratic residues mod p
    # and a root for each residue
    residue_roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in residue_roots:
            residue_roots[r] = x
    
    # Precompute a boolean array for O(1) membership test
    is_residue = [False] * (p + 1)
    for r in residue_roots:
        is_residue[r] = True
    
    # Search ascending from s=1 to find the minimum s
    # such that s, s+1, s+2, s+3, s+4 are all nonzero QRs mod p
    # and s+4 <= p-1 = 1008 (no wraparound)
    for s in range(1, p - 4 + 1):  # s from 1 to 1004
        # Check all five values are nonzero quadratic residues
        all_res = True
        for i in range(5):
            if not is_residue[s + i]:
                all_res = False
                break
        if all_res:
            return {'start': s, 'roots': [residue_roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
