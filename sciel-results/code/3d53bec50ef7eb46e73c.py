"""Rewritten program to find the minimum start of a run of 5 quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Map each quadratic residue to its smallest positive root in [1, p-1]
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        # We want the smallest root, so only set if not already present
        # Since we iterate x from 1 upwards, the first time we see a residue r,
        # x is the smallest root for that residue (except for symmetry x and p-x).
        # Actually, x and p-x give the same square.
        # e.g., 1^2 = 1, (1008)^2 = 1.
        # We just need *a* root. The problem doesn't specify which root, just that
        # ri**2 mod 1009 == s+i and 1<=ri<=1008.
        # So any valid root works.
        if r not in roots:
            roots[r] = x
            
    # Search for the minimum start s
    # s ranges from 1 to 1004 (since s+4 <= 1008)
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if all((s + i) in roots for i in range(5)):
            # Extract the roots
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # If no run found (should not happen for p=1009, m=5)
    raise RuntimeError("No run of 5 quadratic residues found")
# EVOLVE-BLOCK-END
