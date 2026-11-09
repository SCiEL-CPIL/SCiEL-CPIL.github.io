"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_set[u].add(v)
        adj_set[v].add(u)
        
    # Check if host is Ramsey (no K4, no Ind-4)
    # For n=17, we can just check all 4-combinations
    # There are 2380 of them.
    is_ramsey = True
    for combo in itertools.combinations(range(n), 4):
        # Check K4
        is_k4 = True
        for i in range(4):
            for j in range(i+1, 4):
                if combo[j] not in adj_set[combo[i]]:
                    is_k4 = False
                    break
            if not is_k4:
                break
        if is_k4:
            is_ramsey = False
            break
            
        # Check Ind-4
        is_ind4 = True
        for i in range(4):
            for j in range(i+1, 4):
                if combo[j] in adj_set[combo[i]]:
                    is_ind4 = False
                    break
            if not is_ind4:
                break
        if is_ind4:
            is_ramsey = False
            break
            
    best_count = 0
    best_vertices = None
    
    # Iterate over all subsets of size >= min_v
    for mask in range(1 << n):
        k = mask.bit_count()
        if k < min_v:
            continue
            
        if not is_ramsey:
            # Need to check validity
            # Check if mask contains any K4 or Ind4
            # This is expensive. 
            # We can precompute all K4 and Ind4 masks in the host.
            # But we broke out of the loop above if found one.
            # If not is_ramsey, we should have collected them.
            # Let's re-do this logic properly if not is_ramsey.
            pass # Placeholder, will implement below if needed
            
        # Calculate degrees
        degrees = set()
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            deg = (adj_mask[i] & mask).bit_count()
            degrees.add(deg)
            m ^= lsb
            
        dc = len(degrees)
        if dc > best_count:
            best_count = dc
            verts = []
            mm = mask
            while mm:
                lsb = mm & (-mm)
                i = lsb.bit_length() - 1
                verts.append(i)
                mm ^= lsb
            best_vertices = verts
            
    if best_vertices is None:
        return list(range(min_v))
        
    return list(best_vertices)
# EVOLVE-BLOCK-END
