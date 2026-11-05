def construct():
    p = 1009
    
    # is_residue[i] is True if i is a non-zero quadratic residue mod p
    is_residue = [False] * p
    # root_map[i] stores a root r such that r*r % p == i, for non-zero residues
    root_map = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_residue[r] = True
            if root_map[r] == 0:
                root_map[r] = x
    
    # We need s, s+1, s+2, s+3, s+4 to all be non-zero quadratic residues.
    # s must be between 1 and 1004 (so s+4 <= 1008).
    # We minimize s, so we iterate from 1 upwards.
    
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # Note: s+i must be < p (1009) and != 0.
        # Since s <= 1004, s+4 <= 1008 < 1009.
        # Since s >= 1, s+i >= 1.
        
        if (is_residue[s] and 
            is_residue[s+1] and 
            is_residue[s+2] and 
            is_residue[s+3] and 
            is_residue[s+4]):
            
            return {
                'start': s,
                'roots': [
                    root_map[s],
                    root_map[s+1],
                    root_map[s+2],
                    root_map[s+3],
                    root_map[s+4]
                ]
            }
            
    # Fallback, though theoretically one should exist
    return {'start': 1004, 'roots': [root_map[1004], root_map[1005], root_map[1006], root_map[1007], root_map[1008]]}
