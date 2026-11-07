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
    
    # Precompute which subsets are cliques or independent sets
    is_valid = [False] * (1 << n)
    is_valid[0] = True
    
    for mask in range(1, 1 << n):
        # Check if mask is a clique
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
            is_valid[mask] = True
            continue
        
        # Check if mask is an independent set
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
            is_valid[mask] = True

    # DP to find minimum number of parts
    INF = 100
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    parts = []
    mask = (1 << n) - 1
    while mask:
        sub = parent[mask]
        vertices = []
        m = sub
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            vertices.append(v)
            m ^= lb
        parts.append([int(v) for v in vertices])
        mask ^= sub
    
    return parts
# EVOLVE-BLOCK-END
