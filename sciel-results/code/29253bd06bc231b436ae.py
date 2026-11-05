import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    p = 1009

    # Precompute: for each non-zero residue, store one root
    # is_residue[v] is True if v is a non-zero quadratic residue mod p
    # root_map[v] is a root r such that r*r % p == v
    is_residue = [False] * p
    root_map = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_residue[r] = True
            if root_map[r] == 0:
                root_map[r] = x

    # Search for the minimum s such that s, s+1, s+2, s+3, s+4 are all non-zero QRs
    # s ranges from 1 to 1004 (since s+4 <= 1008)
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback: return best found or raise if none
            # Given the density of QRs (~50%), a run of 5 is highly likely to exist early.
            # If we hit the deadline, it implies a pathological case or bug, but we handle it gracefully.
            break
        
        if (is_residue[s] and 
            is_residue[s+1] and 
            is_residue[s+2] and 
            is_residue[s+3] and 
            is_residue[s+4]):
            
            roots = [
                root_map[s],
                root_map[s+1],
                root_map[s+2],
                root_map[s+3],
                root_map[s+4]
            ]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
