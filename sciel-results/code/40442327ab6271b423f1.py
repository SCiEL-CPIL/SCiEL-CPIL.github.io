"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    
    # Adjacency as sets
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Bitmask adjacency for fast clique/independent checks
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # Non-adjacency mask (including self) for independent set checks
    full_mask = (1 << n) - 1
    non_adj_mask = [full_mask ^ adj_mask[i] for i in range(n)]
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    
    # Base case: empty set is both
    is_clique[0] = True
    is_independent[0] = True
    
    # Single vertices are both
    for v in range(n):
        is_clique[1 << v] = True
        is_independent[1 << v] = True
    
    # Compute for all masks
    for mask in range(1, 1 << n):
        # Find lowest set bit
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        
        # Check if v is adjacent to all in rest
        if (adj_mask[v] & rest) == rest:
            is_clique[mask] = is_clique[rest]
        else:
            is_clique[mask] = False
        
        # Check if v is non-adjacent to all in rest
        if (adj_mask[v] & rest) == 0:
            is_independent[mask] = is_independent[rest]
        else:
            is_independent[mask] = False
    
    # min_parts[mask] = minimum number of parts to partition 'mask'
    INF = float('inf')
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    min_parts_choice = [0] * (1 << n)  # stores the subset s that achieves the minimum
    
    for mask in range(1, 1 << n):
        best = INF
        best_s = 0
        # Iterate over all non-empty subsets s of mask
        s = mask
        while s:
            if is_clique[s] or is_independent[s]:
                val = 1 + min_parts[mask ^ s]
                if val < best:
                    best = val
                    best_s = s
            s = (s - 1) & mask
        
        min_parts[mask] = best
        min_parts_choice[mask] = best_s
    
    # The answer for the full graph is min_parts[full_mask]
    if min_parts[full_mask] == INF:
        # Should not happen, fallback
        return [[v] for v in range(n)]
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        s = min_parts_choice[mask]
        # Convert s to list of vertices
        verts = [v for v in range(n) if (s >> v) & 1]
        parts.append(verts)
        mask ^= s
    
    # Ensure all parts are non-empty and return as list of lists
    final = []
    for p in parts:
        if p:
            final.append(list(p))
    
    return final

# EVOLVE-BLOCK-END
