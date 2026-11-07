import sys
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_independent for all masks
    # is_clique[mask] is True if mask is a clique
    # is_indep[mask] is True if mask is an independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest bit
        lb = mask & (-mask)
        u = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Recurrence for clique: mask is clique iff rest is clique AND u is connected to all in rest
        # u connected to all in rest means (rest & ~adj[u]) == 0
        if is_clique[rest] and (rest & ~adj[u]) == 0:
            is_clique[mask] = True
        else:
            is_clique[mask] = False
            
        # Recurrence for independent set: mask is indep iff rest is indep AND u is NOT connected to any in rest
        # u not connected to any in rest means (rest & adj[u]) == 0
        if is_indep[rest] and (rest & adj[u]) == 0:
            is_indep[mask] = True
        else:
            is_indep[mask] = False

    def valid(mask):
        return is_clique[mask] or is_indep[mask]
        
    # IDDFS
    # Try k from 1 to n
    # Vertices to assign: 0 to n-1
    # We can sort vertices by some heuristic, e.g., degree descending
    # But for simplicity, let's stick to 0..n-1 first. If slow, add heuristic.
    
    # To speed up, we can precompute for each vertex v and each mask M,
    # whether M U {v} is valid.
    # valid_with_v[v][mask] = valid(mask | (1<<v))
    # This is 14 * 16384 booleans, very small.
    
    valid_add = [[False] * (1 << n) for _ in range(n)]
    for v in range(n):
        bit = 1 << v
        for mask in range(1 << n):
            if mask & bit:
                valid_add[v][mask] = False # v already in mask, shouldn't happen in our DFS
            else:
                valid_add[v][mask] = valid(mask | bit)
                
    def dfs(idx, parts, max_parts):
        """
        idx: index of next vertex to assign (0 to n)
        parts: list of masks representing current parts
        max_parts: maximum allowed parts
        Returns True if a valid partition exists, and updates parts if so.
        """
        if idx == n:
            return True
            
        v = idx
        bit = 1 << v
        
        # Try to add v to existing parts
        for i in range(len(parts)):
            if valid_add[v][parts[i]]:
                parts[i] |= bit
                if dfs(idx + 1, parts, max_parts):
                    return True
                parts[i] ^= bit # backtrack
                
        # Try to start a new part
        if len(parts) < max_parts:
            parts.append(bit)
            if dfs(idx + 1, parts, max_parts):
                return True
            parts.pop()
            
        return False

    for k in range(1, n + 1):
        # Heuristic: Start with empty parts
        # To break symmetry, we can force the first part to contain vertex 0?
        # Actually, the DFS above will try adding v=0 to existing parts (none), then start new part.
        # So part 0 will always contain vertex 0.
        # This is a good symmetry breaker.
        
        # Optimization: If k is small, we can pre-check if it's possible?
        # Not necessary for n=14.
        
        parts = []
        if dfs(0, parts, k):
            # Convert parts (masks) to lists of vertices
            result = []
            for mask in parts:
                vertices = []
                m = mask
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    vertices.append(v)
                    m ^= lb
                result.append(vertices)
            return result
            
    # Should not reach here
    return [[i] for i in range(n)]
