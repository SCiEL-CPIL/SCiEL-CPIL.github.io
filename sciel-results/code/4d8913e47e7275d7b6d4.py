"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for u, v in P['edges']:
        edges.add((u, v))
        edges.add((v, u))
    
    # Precompute adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute validity of all subsets
    # valid[mask] = True if mask is a clique or an independent set
    valid = [False] * (1 << n)
    valid[0] = True # Empty set is technically both, but we won't use it as a part
    
    for mask in range(1, 1 << n):
        # Check clique
        is_clique = True
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            rest = mask ^ lb
            if rest & ~adj[u]:
                is_clique = False
                break
            m ^= lb
        
        if is_clique:
            valid[mask] = True
            continue
            
        # Check independent set
        is_independent = True
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            rest = mask ^ lb
            if rest & adj[u]:
                is_independent = False
                break
            m ^= lb
            
        if is_independent:
            valid[mask] = True

    full_mask = (1 << n) - 1
    
    # Iterative Deepening DFS
    # We search for a partition into k parts
    # k starts from 1
    # Lower bound: 1 if valid[full_mask] else 2? 
    # Actually, just start from 1.
    
    solution = []
    
    def dfs(rem, k):
        if k == 0:
            return rem == 0
        if rem == 0:
            return k == 0
        if k == 1:
            if valid[rem]:
                solution.append(rem)
                return True
            else:
                return False
        
        # Find the smallest vertex in rem
        # rem is non-zero
        v_min_bit = rem & (-rem)
        
        # Iterate through submasks of rem that contain v_min_bit
        # Let rest = rem ^ v_min_bit
        # v = sub | v_min_bit where sub is a submask of rest
        rest = rem ^ v_min_bit
        sub = rest
        while True:
            v = sub | v_min_bit
            if valid[v]:
                solution.append(v)
                if dfs(rem ^ v, k - 1):
                    return True
                solution.pop()
            
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
        return False

    for k in range(1, n + 1):
        solution.clear()
        if dfs(full_mask, k):
            # Convert masks to lists of vertices
            result = []
            for mask in solution:
                verts = []
                m = mask
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    verts.append(int(v))
                    m ^= lb
                result.append(verts)
            return result
            
    return []
# EVOLVE-BLOCK-END
