import time

def construct():
    p = 1009
    # We need to find the smallest s in [1, 1004] such that 
    # s, s+1, s+2, s+3, s+4 are all quadratic residues mod p.
    # And s+i must be in [1, p-1] for all i in 0..4.
    # This implies s >= 1 and s+4 <= p-1 => s <= p-5 = 1004.
    
    # 1. Precompute quadratic residues and their roots
    # roots[r] = x such that x*x % p == r. We just need one root.
    roots = {}
    is_residue = [False] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_residue[r] = True
            # Store the first root we find for each residue.
            # The task requires returning *a* root, not necessarily the smallest.
            if r not in roots:
                roots[r] = x

    # 2. Search for the smallest s
    # We iterate s from 1 to 1004.
    # For each s, check if s, s+1, s+2, s+3, s+4 are all in is_residue.
    
    # Optimization: Use a bitset or just check the boolean array.
    # Given p is small, a simple loop is sufficient.
    
    for s in range(1, p - 4): # s goes up to 1004
        # Check s, s+1, s+2, s+3, s+4
        if (is_residue[s] and 
            is_residue[s+1] and 
            is_residue[s+2] and 
            is_residue[s+3] and 
            is_residue[s+4]):
            
            # Found the smallest s
            # Construct the list of roots
            res_roots = [
                roots[s],
                roots[s+1],
                roots[s+2],
                roots[s+3],
                roots[s+4]
            ]
            
            return {'start': s, 'roots': res_roots}
            
    # If no run is found (should not happen for p=1009), raise error
    raise RuntimeError("No valid run of 5 consecutive quadratic residues found")
