import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    # Popcount lookup for sorting
    popcount = [0] * size
    for mask in range(1, size):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # v connected to all in rest?
        is_clique[mask] = is_clique[rest] and (adj_mask[v] & rest) == rest
        # v connected to none in rest?
        is_independent[mask] = is_independent[rest] and (adj_mask[v] & rest) == 0
    
    # For each vertex, list valid masks containing it, sorted by size descending
    valid_masks_for_vertex = [[] for _ in range(n)]
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                valid_masks_for_vertex[v].append(mask)
                m ^= lsb
    
    for v in range(n):
        valid_masks_for_vertex[v].sort(key=lambda x: popcount[x], reverse=True)
    
    # Try k from 1 upward
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        found = [False]
        best_partition = [None]
        
        def backtrack(remaining, parts):
            if time.time() > deadline:
                return False
            if found[0]:
                return True
            
            if remaining == 0:
                found[0] = True
                best_partition[0] = [list(p) for p in parts]
                return True
            
            # Pick vertex with fewest valid mask options from remaining
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Try all valid masks containing v that are subsets of remaining
            for S in valid_masks_for_vertex[v]:
                if S & remaining != S:
                    continue
                if popcount[S] > popcount[remaining]:
                    continue
                
                # Extract vertices from S
                part_verts = []
                m = S
                while m:
                    lsb_s = m & -m
                    u = lsb_s.bit_length() - 1
                    part_verts.append(u)
                    m ^= lsb_s
                
                parts.append(part_verts)
                if backtrack(remaining ^ S, parts):
                    return True
                parts.pop()
            
            return False
        
        if backtrack(full_mask, []):
            if best_partition[0] is not None:
                return best_partition[0]
    
    # Fallback: greedy
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts
