import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask representation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_independent for all masks
    # mask is an integer 0 to 2^n - 1
    # is_clique[mask] is True if vertices in mask form a clique
    # is_independent[mask] is True if vertices in mask form an independent set
    
    # Efficient check:
    # A set S is a clique iff for all v in S, (S \ {v}) is subset of adj[v]
    # A set S is an independent set iff for all v in S, (S \ {v}) is subset of non_adj[v]
    
    non_adj = [0] * n
    all_mask = (1 << n) - 1
    for v in range(n):
        non_adj[v] = all_mask ^ adj[v] ^ (1 << v) # Exclude self

    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: rest must be subset of adj[v]
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        # Check independent: rest must be subset of non_adj[v]
        if is_independent[rest] and (rest & ~non_adj[v]) == 0:
            is_independent[mask] = True
            
    valid_part = [False] * (1 << n)
    for mask in range(1 << n):
        valid_part[mask] = is_clique[mask] or is_independent[mask]

    # Find max size of a valid part for pruning
    max_part_size = 0
    for mask in range(1 << n):
        if valid_part[mask]:
            size = bin(mask).count('1')
            if size > max_part_size:
                max_part_size = size

    def dfs(unassigned, parts, k_limit):
        # parts is a list of masks
        # unassigned is a bitmask of vertices not yet assigned
        if unassigned == 0:
            return True
            
        # Pruning: Can we fit remaining vertices into remaining parts?
        remaining_parts = k_limit - len(parts)
        if remaining_parts < 0:
            return False
            
        unassigned_count = bin(unassigned).count('1')
        if unassigned_count > remaining_parts * max_part_size:
            return False
            
        # Pick a vertex to assign. Heuristic: pick one with fewest options? 
        # For simplicity, pick lowest set bit.
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        new_unassigned = unassigned ^ lsb
        
        # Try adding v to an existing part
        # To break symmetry, we can try parts in a specific order.
        # We iterate through existing parts.
        for i in range(len(parts)):
            p_mask = parts[i]
            new_p_mask = p_mask | lsb
            if valid_part[new_p_mask]:
                parts[i] = new_p_mask
                if dfs(new_unassigned, parts, k_limit):
                    return True
                parts[i] = p_mask # backtrack
                
        # Try creating a new part
        if len(parts) < k_limit:
            parts.append(lsb)
            if dfs(new_unassigned, parts, k_limit):
                return True
            parts.pop() # backtrack
            
        return False

    # Search for minimum k
    for k in range(1, n + 1):
        # Initial parts: empty list
        # We need to ensure we don't timeout. 
        # Given n=14, this should be fast.
        # To be safe, we can add a time check, but likely not needed.
        if dfs((1 << n) - 1, [], k):
            # Reconstruct parts
            # We need to run dfs again to get the parts, or modify dfs to store the solution
            # Let's modify dfs to return the solution or store it in a global
            pass
            
    # The above loop doesn't capture the solution. 
    # Let's rewrite to store the solution when found.
    
    def dfs_solve(unassigned, parts, k_limit):
        if unassigned == 0:
            return True
        remaining_parts = k_limit - len(parts)
        if remaining_parts < 0:
            return False
        unassigned_count = bin(unassigned).count('1')
        if unassigned_count > remaining_parts * max_part_size:
            return False
            
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        new_unassigned = unassigned ^ lsb
        
        for i in range(len(parts)):
            p_mask = parts[i]
            new_p_mask = p_mask | lsb
            if valid_part[new_p_mask]:
                parts[i] = new_p_mask
                if dfs_solve(new_unassigned, parts, k_limit):
                    return True
                parts[i] = p_mask
                
        if len(parts) < k_limit:
            parts.append(lsb)
            if dfs_solve(new_unassigned, parts, k_limit):
                return True
            parts.pop()
            
        return False

    for k in range(1, n + 1):
        parts = []
        if dfs_solve((1 << n) - 1, parts, k):
            # Convert masks to lists of vertices
            result = []
            for mask in parts:
                vertices = []
                for v in range(n):
                    if mask & (1 << v):
                        vertices.append(v)
                if vertices: # Ensure non-empty
                    result.append(vertices)
            return result
            
    # Fallback: Greedy (should not happen)
    es = {tuple(e) for e in P['edges']}
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

# EVOLVE-BLOCK-END
