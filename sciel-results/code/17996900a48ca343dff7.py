import time

def construct():
    """
    Finds the minimum start s for a run of 5 consecutive non-zero quadratic residues mod 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    p = 1009
    m = 5
    
    # Precompute roots for all non-zero quadratic residues
    # We only need one root per residue to satisfy the interface.
    root_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in root_map:
            root_map[r] = x
            
    # Search for the minimum s in ascending order
    # s ranges from 1 to p - m (1004) to ensure s+m-1 < p
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+4 are all in root_map
        # Since s <= 1004 and m=5, max index is 1008, which is < 1009, so no modulo wrap-around needed for indices
        roots = []
        valid = True
        for i in range(m):
            val = s + i
            if val in root_map:
                roots.append(root_map[val])
            else:
                valid = False
                break
        
        if valid:
            return {'start': s, 'roots': roots}
            
    # Fallback, though mathematically a solution is guaranteed to exist for p=1009, m=5
    raise RuntimeError("No valid run found")
