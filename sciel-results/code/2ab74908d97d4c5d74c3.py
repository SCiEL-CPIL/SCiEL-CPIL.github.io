def construct():
    """
    Find the minimum start s for 5 consecutive quadratic residues mod 1009.
    """
    p = 1009
    # Precompute the set of non-zero quadratic residues and their roots
    residues = set()
    roots = {}
    
    # Iterate through all possible bases x
    for x in range(1, p):
        r = (x * x) % p
        # Exclude zero residues
        if r == 0:
            continue
        # Add to set for O(1) lookup
        residues.add(r)
        # Store the first root found for each residue
        if r not in roots:
            roots[r] = x
    
    # Search for the smallest s
    # s must be in [1, 1004]
    # s+i must be in [1, 1008]
    # Since s >= 1, s+i >= 1.
    # Since s <= 1004, s+4 <= 1008.
    # We need s, s+1, s+2, s+3, s+4 to all be in `residues`.
    
    for s in range(1, 1005):
        # Check if the window [s, s+4] consists entirely of quadratic residues
        if all((s + i) in residues for i in range(5)):
            # Construct the result
            # Get the roots for s, s+1, s+2, s+3, s+4
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # If no run is found (should not happen for p=1009 with m=5)
    raise RuntimeError("No valid run found")
