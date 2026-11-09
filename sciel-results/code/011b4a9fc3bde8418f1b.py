def construct():
    p = 1009
    # is_residue[i] will be True if i is a quadratic residue mod p
    is_residue = [False] * p
    # roots[i] will store one root x such that x^2 == i mod p
    # We use 0 to indicate "not a residue" or "no root stored yet"
    # Since roots must be 1..1008, 0 is a safe sentinel for "none".
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        is_residue[r] = True
        # Store the first root found. Since we iterate x from 1, 
        # this will be the smaller of the two roots (x, p-x) if x < p/2,
        # or just a valid root. It doesn't matter which valid root we pick.
        if roots[r] == 0:
            roots[r] = x
            
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    # s must be at least 1 and s+4 must be at most 1008 (since 1009 is 0 mod p and excluded)
    # So s can go up to 1004.
    for s in range(1, 1005):
        # Check if all 5 consecutive numbers are quadratic residues
        if (is_residue[s] and is_residue[s+1] and is_residue[s+2] and 
            is_residue[s+3] and is_residue[s+4]):
            
            # Extract the roots
            res_roots = [roots[s+i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # This should never happen for p=1009, but just in case
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
