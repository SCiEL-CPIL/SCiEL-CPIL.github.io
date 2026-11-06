"""Find the minimum start s such that s, s+1, ..., s+4 are all quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a set of all quadratic residues mod p (excluding 0)
    # For each x in 1..p-1, x^2 mod p is a quadratic residue
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Also store one root for each residue
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # Search for the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all in residues
    # and none of them is 0 (zero residues are excluded)
    for s in range(1, p - 4):  # s+4 <= 1008, so s <= 1004
        valid = True
        for i in range(5):
            val = s + i
            if val == 0 or val not in residues:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
