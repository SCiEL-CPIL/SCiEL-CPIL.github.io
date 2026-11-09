import time

def construct():
    n = 14
    edges = P['edges']
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Bitmasks for adjacency
    adj_mask = [0] * n
    non_adj_mask = [0] * n
    all_mask = (1 << n) - 1
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
            else:
                non_adj_mask[u] |= (1 << v)
        
    # Precompute masks for clique/independent checks
    # For a set S, is_clique(S, v) iff (adj_mask[v] & S) == S
    # For a set S, is_independent(S, v) iff (non_adj_mask[v] & S) == S
    
    # Reorder vertices by degree (descending) to improve pruning
    degrees = [sum(1 for v in range(n) if adj[u][v]) for u in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    pos = {v: i for i, v in enumerate(order)}
    
    # Remap adjacency to new order? Or just use original indices in masks.
    # Using original indices is fine if we iterate through 'order'.
    
    def search(idx, parts, types, k):
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try assigning v to each part j in 0..k-1
        for j in range(k):
            if not parts[j]:
                # Symmetry breaking: only use the first empty part
                if j > 0 and not parts[j-1]:
                    continue
                
                # New part is [v], type is flexible (3)
                parts[j] = [v]
                types[j] = 3
                # Mask for this part: 1 << v
                part_masks[j] = 1 << v
                
                if search(idx + 1, parts, types, k):
                    return True
                
                parts[j] = []
                types[j] = 0
                part_masks[j] = 0
            else:
                t = types[j]
                mask = part_masks[j]
                
                if t == 3: # Flexible (size 1)
                    u = parts[j][0]
                    # Can become clique or independent
                    # Since (u,v) is either edge or not, it's always valid
                    # But we need to update type
                    new_type = 1 if adj[u][v] else 2
                    parts[j].append(v)
                    types[j] = new_type
                    part_masks[j] = mask | (1 << v)
                    
                    if search(idx + 1, parts, types, k):
                        return True
                    
                    parts[j].pop()
                    types[j] = 3
                    part_masks[j] = mask
                elif t == 1: # Clique
                    # Check if v is adjacent to all in part
                    # (adj_mask[v] & mask) == mask
                    if (adj_mask[v] & mask) == mask:
                        parts[j].append(v)
                        part_masks[j] = mask | (1 << v)
                        if search(idx + 1, parts, types, k):
                            return True
                        parts[j].pop()
                        part_masks[j] = mask
                elif t == 2: # Independent
                    # Check if v is NOT adjacent to any in part
                    # (non_adj_mask[v] & mask) == mask
                    if (non_adj_mask[v] & mask) == mask:
                        parts[j].append(v)
                        part_masks[j] = mask | (1 << v)
                        if search(idx + 1, parts, types, k):
                            return True
                        parts[j].pop()
                        part_masks[j] = mask
        
        return False

    # Try k from 1 to n
    for k in range(1, n + 1):
        parts = [[] for _ in range(k)]
        types = [0] * k
        part_masks = [0] * k
        
        if search(0, parts, types, k):
            final_parts = [p for p in parts if p]
            return [list(p) for p in final_parts]
    
    # Fallback (should not happen)
    return [[i] for i in range(n)]

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
