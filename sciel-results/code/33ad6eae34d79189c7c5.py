"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Order vertices by degree descending to prune earlier
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Precompute valid additions?
    # For a given mask and vertex v, is it a clique? is it independent?
    # We will compute on the fly in the recursion.
    
    def dfs(idx, parts):
        """
        parts: list of tuples (mask, is_clique, is_independent)
        """
        if idx == n:
            return True
        
        v = order[idx]
        bit_v = 1 << v
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            mask, is_c, is_i = parts[i]
            
            # Check if v can be added to this part
            # New clique status: current is clique AND v connects to all in mask
            new_is_c = is_c and (mask & ~adj[v] == 0)
            # New independent status: current is independent AND v has no edges to mask
            new_is_i = is_i and (mask & adj[v] == 0)
            
            if new_is_c or new_is_i:
                # Update part i
                parts[i] = (mask | bit_v, new_is_c, new_is_i)
                if dfs(idx + 1, parts):
                    return True
                # Backtrack
                parts[i] = (mask, is_c, is_i)
        
        # Try adding v to a new part
        # Symmetry breaking: we only add to a new part if it's the first empty one?
        # Actually, parts are a list. We don't have "empty" parts in the list, we append.
        # To avoid symmetry, we should not add a new part if we have already skipped an empty slot?
        # Since we append, the "new part" is always at the end.
        # The symmetry is: Part A and Part B are both empty. Putting v in A vs B is same.
        # Since we only add to the end, and we don't have pre-allocated empty parts, 
        # this specific symmetry is handled by the fact that we don't have "holes".
        # However, if we have parts [P1, P2] and we add P3, it's distinct from adding P4? No, P3 is the new one.
        # The symmetry is between the *content* of parts.
        # Standard trick: The first vertex must go to part 0.
        # Subsequent vertices: If we place v in a new part, we must ensure we don't create a part that is a subset of a previous one? No.
        # Simple symmetry breaking: 
        # 1. The first vertex (idx=0) must go to a new part (which becomes part 0).
        # 2. For subsequent vertices, we can try existing parts.
        # 3. We can try a new part, BUT we should only do so if we haven't already tried placing it in a way that creates symmetry.
        # A common constraint: If we create a new part, the new part must contain the current vertex v. 
        # To break symmetry between different "new parts", we can say: 
        # If we are creating a new part, we must do so only if we have not created a part with the same "signature" before?
        # Simpler: Just allow it. For n=14, the branching factor is manageable if k is small.
        
        if len(parts) < k:
            parts.append((bit_v, True, True))
            if dfs(idx + 1, parts):
                return True
            parts.pop()
            
        return False

    # Iterate k from 1 to n
    for k in range(1, n + 1):
        # Initial state: no parts
        # We can force the first vertex to go to a new part to reduce symmetry
        # But dfs handles idx=0. If parts is empty, it will go to "new part" logic.
        # Let's start with empty parts.
        if dfs(0, []):
            # Reconstruct the partition from the final parts
            # The parts are (mask, is_c, is_i)
            # We need to return a list of lists of vertices
            result = []
            # We need to capture the final state. 
            # The dfs above modifies a list. We need to store the successful path.
            # Let's refactor dfs to return the parts or use a global variable.
            pass
            
    # Refactored approach to capture the solution
    solution_parts = []
    
    def dfs_capture(idx, parts):
        if idx == n:
            solution_parts.extend([p[0] for p in parts])
            return True
        
        v = order[idx]
        bit_v = 1 << v
        
        for i in range(len(parts)):
            mask, is_c, is_i = parts[i]
            new_is_c = is_c and (mask & ~adj[v] == 0)
            new_is_i = is_i and (mask & adj[v] == 0)
            
            if new_is_c or new_is_i:
                parts[i] = (mask | bit_v, new_is_c, new_is_i)
                if dfs_capture(idx + 1, parts):
                    return True
                parts[i] = (mask, is_c, is_i)
        
        if len(parts) < k:
            parts.append((bit_v, True, True))
            if dfs_capture(idx + 1, parts):
                return True
            parts.pop()
            
        return False

    for k in range(1, n + 1):
        solution_parts.clear()
        if dfs_capture(0, []):
            # Convert masks to lists of vertices
            final_parts = []
            for mask in solution_parts:
                part = []
                m = mask
                while m:
                    v_bit = m & (-m)
                    v = v_bit.bit_length() - 1
                    part.append(v)
                    m ^= v_bit
                final_parts.append(part)
            return final_parts

    # Fallback (should not happen)
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
