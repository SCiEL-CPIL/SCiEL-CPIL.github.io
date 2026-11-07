"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 85.0
    
    # Build adjacency list and bitmask neighbors
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_independent for all subsets
    # 2^14 = 16384 subsets, very fast
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be adjacent to all in rest
        # And rest must be a clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: v must NOT be adjacent to any in rest
        # And rest must be independent
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    # Order vertices by degree (descending) to fail faster? 
    # Or maybe ascending? 
    # Let's try a few orderings.
    # Standard: 0..n-1
    # Degree based: sorted(range(n), key=lambda x: -bin(adj[x]).count('1'))
    
    orderings = [
        list(range(n)),
        sorted(range(n), key=lambda x: -bin(adj[x]).count('1')),
        sorted(range(n), key=lambda x: bin(adj[x]).count('1'))
    ]
    
    # We search for the minimum k
    for k in range(1, n + 1):
        for order in orderings:
            if time.time() > deadline:
                return None # Should not happen, but safe
            
            parts = [0] * k  # Bitmasks for each part
            # parts[i] is the mask of vertices in part i
            
            # Symmetry breaking:
            # We assign vertices in the order defined by 'order'.
            # We maintain an index 'next_part' which is the index of the first empty part.
            # We can only add a vertex to an existing non-empty part, or to 'next_part'.
            # This ensures parts are filled in order and avoids permutations of parts.
            
            def bt(idx, next_part):
                if time.time() > deadline:
                    return None
                
                if idx == n:
                    # Check if all parts are non-empty
                    # Actually, since we only create parts up to next_part, 
                    # and we ensure we don't skip empty parts, 
                    # we just need to ensure we used exactly k parts?
                    # No, we are searching for a partition into AT MOST k parts?
                    # The problem asks for a partition into non-empty lists.
                    # If we find a solution with < k parts, that's also valid for k?
                    # No, we iterate k from 1 upwards. The first valid k is the minimum.
                    # So if we find a valid assignment for k parts, we return it.
                    # We must ensure all k parts are non-empty.
                    for i in range(k):
                        if parts[i] == 0:
                            return None
                    return [i for i in range(k) for _ in range(bin(parts[i]).count('1'))] # Placeholder, need to extract vertices
                    
                v = order[idx]
                v_bit = 1 << v
                
                # Try adding to existing parts 0 .. next_part - 1
                for i in range(next_part):
                    new_mask = parts[i] | v_bit
                    if is_clique[new_mask] or is_indep[new_mask]:
                        parts[i] = new_mask
                        res = bt(idx + 1, next_part)
                        if res is not None:
                            return res
                        parts[i] ^= v_bit # Restore
                        
                # Try adding to the next empty part
                if next_part < k:
                    parts[next_part] = v_bit
                    # v_bit is always a valid clique and independent set
                    res = bt(idx + 1, next_part + 1)
                    if res is not None:
                        return res
                    parts[next_part] = 0
                    
                return None

            # Initial call
            # We need to map the bitmask solution back to vertex lists
            # The bt function returns a list of vertices? No, let's make it return the parts masks
            # and then convert.
            
            def bt_masks(idx, next_part):
                if time.time() > deadline:
                    return None
                
                if idx == n:
                    for i in range(k):
                        if parts[i] == 0:
                            return None
                    return tuple(parts)
                    
                v = order[idx]
                v_bit = 1 << v
                
                for i in range(next_part):
                    new_mask = parts[i] | v_bit
                    if is_clique[new_mask] or is_indep[new_mask]:
                        parts[i] = new_mask
                        res = bt_masks(idx + 1, next_part)
                        if res is not None:
                            return res
                        parts[i] ^= v_bit
                        
                if next_part < k:
                    parts[next_part] = v_bit
                    res = bt_masks(idx + 1, next_part + 1)
                    if res is not None:
                        return res
                    parts[next_part] = 0
                    
                return None

            res_masks = bt_masks(0, 0)
            if res_masks is not None:
                # Convert masks to lists of vertices
                partition = []
                for mask in res_masks:
                    verts = []
                    for v in range(n):
                        if mask & (1 << v):
                            verts.append(v)
                    if verts:
                        partition.append(verts)
                return partition

    # Fallback: trivial partition
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
