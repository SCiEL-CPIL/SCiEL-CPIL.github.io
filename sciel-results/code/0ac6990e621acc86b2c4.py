import time

P = 1009
M = 5

def construct():
    start_time = time.time()
    limit = 85.0
    
    # Array to store a valid root for each non-zero quadratic residue
    # root_map[v] = x where x*x % P == v, 0 if v is not a non-zero QR
    root_map = [0] * P
    
    # Build the bitmask where bit i is 1 if i is a non-zero quadratic residue
    residue_mask = 0
    
    # Single pass to compute residues and build mask
    for x in range(1, P):
        sq = (x * x) % P
        if sq != 0:
            if root_map[sq] == 0:
                root_map[sq] = x
            residue_mask |= (1 << sq)
    
    # Target pattern for 5 consecutive set bits
    pattern = (1 << M) - 1
    
    # Search for the smallest s in [1, P-M]
    # s+i must be <= P-1 (1008) to be a valid non-zero residue index in our mask
    # Max s is P - M
    s = 1
    max_s = P - M
    
    while s <= max_s:
        # Check if bits s through s+4 are all set in residue_mask
        # Shift mask right by s, then check the lowest 5 bits
        if ((residue_mask >> s) & pattern) == pattern:
            # Found the minimal s
            # Extract roots for s, s+1, ..., s+4
            result_roots = [root_map[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}
        
        s += 1
        
        # Deadline check
        if time.time() - start_time > limit:
            break
            
    # This should never happen for P=1009, M=5
    raise RuntimeError("Search failed or deadline exceeded")
