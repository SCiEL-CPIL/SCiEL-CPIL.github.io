import time

def construct():
    p = 1009
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60  # 60 seconds limit
    
    # Precompute quadratic residues and one root for each
    # We need to find s such that s, s+1, s+2, s+3, s+4 are residues.
    # The values s+i are in [1, 1008] since 1 <= s <= 1004.
    # So 0 is not involved.
    
    roots = {}
    is_residue = [False] * p
    
    for x in range(1, p):
        r = (x * x) % p
        is_residue[r] = True
        # Store the first root found (smallest x)
        if r not in roots:
            roots[r] = x
            
    # Search for the minimum s
    # We want to minimize s, so we iterate from 1 upwards.
    # The condition is that s, s+1, s+2, s+3, s+4 are all quadratic residues.
    # Max s is 1004 because s+4 <= 1008 (to stay in 1..1008 range and avoid wraparound to 0 or negative in the context of "1<=ri<=1008" and "zero residues excluded").
    # Actually, the problem says 1 <= s <= 1004.
    
    for s in range(1, 1005):
        if time.time() > deadline - 1:  # Safety margin
            break
            
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        if (is_residue[s] and 
            is_residue[s+1] and 
            is_residue[s+2] and 
            is_residue[s+3] and 
            is_residue[s+4]):
            
            # Construct the roots list
            # Ensure roots exist (they do if is_residue is True)
            r_roots = [roots[s+i] for i in range(5)]
            
            # Verify constraints: 1 <= ri <= 1008
            # Since x was in 1..1008, roots are in 1..1008.
            # ri**2 mod 1009 == s+i is guaranteed by construction.
            
            return {'start': s, 'roots': r_roots}
            
    # If no s found (should not happen for p=1009), raise error
    raise RuntimeError('No run found')
