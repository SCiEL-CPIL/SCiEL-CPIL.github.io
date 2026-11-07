import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute vertex order by degree (descending) to prune faster
    # Vertices with high degree are more constrained in independent sets, 
    # but also more constrained in cliques? 
    # Actually, high degree makes it harder to put in an independent set.
    # Low degree makes it harder to put in a clique.
    # Let's just use a static order or degree-sorted.
    order = sorted(range(n), key=lambda v: adj[v].bit_count(), reverse=True)
    
    # Map original vertex index to position in order? 
    # No, we just iterate through `order`.
    
    # Helper to check if a set of vertices (mask) is a clique
    def is_clique(mask):
        if mask == 0: return True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if (adj[v] & mask) != (mask ^ lsb):
                return False
        return True

    # Helper to check if a set of vertices (mask) is an independent set
    def is_independent(mask):
        if mask == 0: return True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if (adj[v] & mask) != 0:
                return False
        return True

    # Backtracking search for a valid partition with at most k parts
    # We want to find if a partition with exactly k parts exists.
    # Actually, we iterate k from 1 to n. The first k that works is the answer.
    
    def search(k):
        # parts: list of [mask, type]
        # type: 0=Clique, 1=Independent, 2=Undetermined
        parts = []
        assigned_mask = 0
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            v_bit = 1 << v
            
            # Try placing v in existing parts
            for i in range(len(parts)):
                mask, ptype = parts[i]
                new_mask = mask | v_bit
                
                # Check if v can be added
                ok = False
                new_ptype = ptype
                
                if ptype == 0: # Clique
                    # v must be adjacent to all in mask
                    if (adj[v] & mask) == mask:
                        ok = True
                        new_ptype = 0
                elif ptype == 1: # Independent
                    # v must be non-adjacent to all in mask
                    if (adj[v] & mask) == 0:
                        ok = True
                        new_ptype = 1
                else: # Undetermined
                    # Check if adding v forces a type or is invalid
                    edges_to_mask = adj[v] & mask
                    non_edges_to_mask = mask & ~adj[v] & ~v_bit # mask doesn't have v_bit yet
                    
                    if edges_to_mask == 0:
                        # v is non-adjacent to all. Can be Independent.
                        # Can it be Clique? Only if mask is empty.
                        if mask == 0:
                            ok = True
                            new_ptype = 2 # Still undetermined? Or force?
                            # If mask is empty, it's just {v}. It can be either.
                            # Let's keep it undetermined to allow flexibility for future vertices.
                        else:
                            # If mask is not empty, and v has no edges to mask,
                            # it cannot be a clique (unless mask was empty).
                            # So it MUST be independent.
                            ok = True
                            new_ptype = 1
                    elif non_edges_to_mask == 0:
                        # v is adjacent to all.
                        if mask == 0:
                            ok = True
                            new_ptype = 2
                        else:
                            # Must be clique.
                            ok = True
                            new_ptype = 0
                    else:
                        # v has some edges and some non-edges to mask.
                        # Cannot be clique or independent.
                        ok = False
                    
                if ok:
                    parts[i] = [new_mask, new_ptype]
                    if backtrack(idx + 1):
                        return True
                    parts[i] = [mask, ptype]
            
            # Try placing v in a new part
            if len(parts) < k:
                # Symmetry breaking: 
                # We only create a new part if we haven't already created a new part 
                # for a previous vertex in this branch that resulted in an empty slot?
                # Actually, since we always append, the "new part" is always at the end.
                # The only symmetry is permuting the existing parts.
                # Since we try all existing parts, and then the new part,
                # we might explore symmetric branches.
                # However, for N=14, this is usually fine.
                
                parts.append([v_bit, 2])
                if backtrack(idx + 1):
                    return True
                parts.pop()
            
            return False
        
        return backtrack(0), parts

    # Iterate k from 1 to n
    # Note: The search function above modifies `parts` in place. 
    # We need to be careful with state restoration or pass a copy.
    # The current implementation restores state on failure, so it's safe.
    # But it returns the `parts` list which is mutated. 
    # If it returns True, `parts` contains the solution.
    
    for k in range(1, n + 1):
        found, parts = search(k)
        if found:
            # Convert parts to list of lists of vertices
            result = []
            for mask, ptype in parts:
                verts = []
                for i in range(n):
                    if mask & (1 << i):
                        verts.append(i)
                result.append(verts)
            
            # Validation
            for part in result:
                mask = 0
                for v in part:
                    mask |= (1 << v)
                if not (is_clique(mask) or is_independent(mask)):
                    # Should not happen
                    pass
            
            return result
            
    # Fallback (should not reach here)
    return [[i] for i in range(n)]
