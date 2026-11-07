import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix and bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_independent for all masks
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        # Find a set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        # v must be connected to all in rest
        if is_clique[rest]:
            # Check if v is connected to all vertices in rest
            # adj[v] & rest should equal rest
            if (adj[v] & rest) == rest:
                is_clique[mask] = True
                
        # Check independent
        # v must NOT be connected to any in rest
        if is_independent[rest]:
            # Check if v is NOT connected to any vertex in rest
            # adj[v] & rest should be 0
            if (adj[v] & rest) == 0:
                is_independent[mask] = True
                
    # Memoization for solve
    memo = {}
    
    def popcount(x):
        return bin(x).count('1')
        
    def solve(rem, k_rem):
        if rem == 0:
            return [] if k_rem == 0 else None
        if k_rem == 0:
            return None
        if (rem, k_rem) in memo:
            return memo[(rem, k_rem)]
            
        pc = popcount(rem)
        if pc < k_rem:
            memo[(rem, k_rem)] = None
            return None
        if pc == k_rem:
            # Each part is a single vertex
            parts = [1 << i for i in range(n) if (rem >> i) & 1]
            memo[(rem, k_rem)] = parts
            return parts
        if k_rem == 1:
            if is_clique[rem] or is_independent[rem]:
                memo[(rem, k_rem)] = [rem]
                return [rem]
            memo[(rem, k_rem)] = None
            return None
            
        # Find smallest vertex in rem
        lsb = rem & -rem
        v = lsb.bit_length() - 1
        rest = rem ^ lsb
        
        # Iterate submasks of rest
        sub = rest
        while True:
            s = sub | lsb
            if is_clique[s] or is_independent[s]:
                res = solve(rem ^ s, k_rem - 1)
                if res is not None:
                    memo[(rem, k_rem)] = [s] + res
                    return [s] + res
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
        memo[(rem, k_rem)] = None
        return None

    all_vertices = (1 << n) - 1
    for k in range(1, n + 1):
        parts = solve(all_vertices, k)
        if parts is not None:
            # Convert bitmasks to lists of vertices
            final_parts = []
            for p in parts:
                vertices = []
                for i in range(n):
                    if (p >> i) & 1:
                        vertices.append(i)
                final_parts.append(vertices)
            return final_parts
            
    # Fallback (should not happen)
    return [[i] for i in range(n)]
