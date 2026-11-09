import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute adjacency masks
    adj_mask = [0] * n
    for a, b in P['edges']:
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
    # Precompute is_clique and is_indep for all masks
    # size = 2^n
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent
        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True
            
    def valid_part(mask):
        return mask == 0 or is_clique[mask] or is_indep[mask]
        
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Helper to pick the most constrained vertex
    def pick_vertex(remaining_mask, parts):
        best_v = -1
        min_options = float('inf')
        
        # Iterate through bits in remaining_mask
        m = remaining_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            options = 0
            # Check if v can be added to any existing part
            for p in parts:
                if valid_part(p | lsb):
                    options += 1
            
            # If options is 0, it must go to a new part.
            # We treat "new part" as an option if we haven't hit the limit, 
            # but for heuristic purposes, 0 existing options is the most constrained.
            
            if options < min_options:
                min_options = options
                best_v = v
                
        return best_v
        
    def backtrack(remaining_mask, parts, max_parts):
        if time.time() > deadline:
            return None
            
        if remaining_mask == 0:
            return parts[:]
            
        v = pick_vertex(remaining_mask, parts)
        if v == -1:
            return None
            
        v_bit = 1 << v
        new_remaining = remaining_mask ^ v_bit
        
        # Try adding v to existing parts
        # Symmetry breaking: if we have multiple identical empty parts? 
        # We don't keep empty parts. 
        # However, if we have parts that are structurally identical? 
        # For cochromatic, parts are defined by their vertex sets. 
        # Two parts with different vertex sets are not symmetric.
        # So we just try all.
        
        for i in range(len(parts)):
            p = parts[i]
            new_p = p | v_bit
            if valid_part(new_p):
                new_parts = parts[:]
                new_parts[i] = new_p
                res = backtrack(new_remaining, new_parts, max_parts)
                if res is not None:
                    return res
                    
        # Try creating a new part
        if len(parts) < max_parts:
            new_parts = parts + [v_bit]
            res = backtrack(new_remaining, new_parts, max_parts)
            if res is not None:
                return res
                
        return None

    # Try increasing K
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        result_parts = backtrack(full_mask, [], k)
        
        if result_parts is not None:
            # Convert bitmasks to lists of vertices
            partition = []
            for p in result_parts:
                verts = []
                for v in range(n):
                    if (p >> v) & 1:
                        verts.append(v)
                if verts:
                    partition.append(verts)
                    
            # Verify
            edge_set = set()
            for a, b in P['edges']:
                edge_set.add((a, b))
                edge_set.add((b, a))
                
            for part in partition:
                is_clique = True
                is_indep = True
                for i in range(len(part)):
                    for j in range(i + 1, len(part)):
                        if (part[i], part[j]) in edge_set:
                            is_indep = False
                        else:
                            is_clique = False
                if not (is_clique or is_indep):
                    raise ValueError("Invalid partition found")
                    
            # Ensure types are correct
            return [list(map(int, part)) for part in partition]
            
    # Fallback
    return [[int(i)] for i in range(n)]
