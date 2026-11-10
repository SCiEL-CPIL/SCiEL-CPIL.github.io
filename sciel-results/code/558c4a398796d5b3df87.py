import time
import random
import sys

sys.setrecursionlimit(10000)

def _is_k4_free(adj):
    """Check if graph with adjacency matrix adj is K4-free."""
    n = len(adj)
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_exact(adj, time_limit=5.0):
    """
    Exact branch and bound to find the largest triangle-free induced subgraph.
    Returns the maximum size.
    """
    n = len(adj)
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    CN = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            CN[a][b] = adj_mask[a] & adj_mask[b]
            CN[b][a] = CN[a][b]
            
    def is_triangle_free(S):
        m = S
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            rest = m ^ lb
            m2 = rest
            while m2:
                lb2 = m2 & (-m2)
                b = lb2.bit_length() - 1
                if (adj_mask[a] >> b) & 1:
                    if CN[a][b] & S:
                        return False
                m2 ^= lb2
            m = rest
        return True

    start_time = time.time()
    best = 0
    # Try sizes from n down to 1
    # Optimization: If we find a triangle-free set of size k, we return k.
    # We iterate masks in descending order of popcount? 
    # Simpler: iterate all masks, track max popcount for triangle-free.
    # 2^18 = 262144.
    # Check each mask.
    for mask in range(1 << n):
        if time.time() - start_time > time_limit:
            break
        # Quick check: if popcount <= best, skip
        if bin(mask).count('1') <= best:
            continue
        if is_triangle_free(mask):
            best = bin(mask).count('1')
            if best == n:
                return n
    return best

def construct():
    deadline = time.time() + 115
    n = 18
    
    # Construction:
    # A: 0-5 (Independent)
    # B: 6-11 (K_{2,2,2})
    # C: 12-17 (K_{2,2,2})
    # Edges:
    # A-B: Complete
    # A-C: Complete
    # B-C: Empty
    # B-internal: K_{2,2,2}
    # C-internal: K_{2,2,2}
    
    adj = [[False] * n for _ in range(n)]
    
    # Part A: 0-5
    # Part B: 6-11
    # Part C: 12-17
    
    # A-B Complete
    for u in range(0, 6):
        for v in range(6, 12):
            adj[u][v] = True
            adj[v][u] = True
            
    # A-C Complete
    for u in range(0, 6):
        for v in range(12, 18):
            adj[u][v] = True
            adj[v][u] = True
            
    # B-Internal K_{2,2,2}
    # Parts: {6,7}, {8,9}, {10,11}
    b_parts = [[6, 7], [8, 9], [10, 11]]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in b_parts[i]:
                for v in b_parts[j]:
                    adj[u][v] = True
                    adj[v][u] = True
                    
    # C-Internal K_{2,2,2}
    # Parts: {12,13}, {14,15}, {16,17}
    c_parts = [[12, 13], [14, 15], [16, 17]]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in c_parts[i]:
                for v in c_parts[j]:
                    adj[u][v] = True
                    adj[v][u] = True
                    
    # Verify K4-free
    if not _is_k4_free(adj):
        # Fallback to K_{6,6,6} if something went wrong (should not happen)
        adj = [[False] * n for _ in range(n)]
        for u in range(0, 6):
            for v in range(6, 12):
                adj[u][v] = True; adj[v][u] = True
            for v in range(12, 18):
                adj[u][v] = True; adj[v][u] = True
        for u in range(6, 12):
            for v in range(12, 18):
                adj[u][v] = True; adj[v][u] = True

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
                
    return edges
