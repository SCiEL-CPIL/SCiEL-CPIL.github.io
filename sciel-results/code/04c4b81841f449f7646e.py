"""Find the minimum start s for 5 consecutive non-zero quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute: for each non-zero residue, store one root
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
    # Search ascending to minimize s
    for s in range(1, p - 4):  # s from 1 to 1004
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
