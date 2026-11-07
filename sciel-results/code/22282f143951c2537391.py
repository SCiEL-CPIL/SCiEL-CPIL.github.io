import time

def construct():
    # Data from the seed program
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all subsets
    # is_clique[mask]: True if mask induces a clique
    # is_ind[mask]: True if mask induces an independent set
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get least significant bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set: v must not be connected to any in rest, and rest must be independent
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    best_parts_count = n
    best_partition = None
    current_parts = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    def search(rem, count):
        nonlocal best_parts_count, best_partition, current_parts
        
        if time.time() > deadline:
            return
        
        if rem == 0:
            if count < best_parts_count:
                best_parts_count = count
                best_partition = [list(part) for part in current_parts]
            return
            
        if count >= best_parts_count:
            return
            
        # Simple lower bound: at least 1 part needed if rem != 0
        if count + 1 > best_parts_count:
            return
            
        # Choose a vertex to anchor the subset
        # Using the first set bit is a standard heuristic
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        # Iterate through all submasks of rem that contain v
        # We can iterate all submasks of rem and filter, or iterate more cleverly.
        # Given n=14, iterating all submasks of rem (2^k) is fine.
        sub = rem
        while sub:
            if (sub & v_bit) and is_valid[sub]:
                # Extract vertices for this subset
                part_verts = []
                temp_sub = sub
                while temp_sub:
                    b = temp_sub & -temp_sub
                    idx = b.bit_length() - 1
                    part_verts.append(idx)
                    temp_sub ^= b
                
                current_parts.append(part_verts)
                search(rem ^ sub, count + 1)
                current_parts.pop()
            
            sub = (sub - 1) & rem
            
    search(full_mask, 0)
    
    # Fallback if search failed or timed out (should not happen for n=14)
    if best_partition is None:
        # Fallback to greedy (like the original)
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

    # Ensure types are correct (int, list)
    # The constructed lists contain ints from bit_length, which are Python ints.
    return best_partition
