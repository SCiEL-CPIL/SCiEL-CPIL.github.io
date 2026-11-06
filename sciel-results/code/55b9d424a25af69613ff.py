import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    N = 16
    
    # Grötzsch graph edges
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 1], [5, 2], [6, 2], [6, 3], [7, 3], [7, 4], 
        [8, 4], [8, 0], [9, 0], [9, 1],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    def build_adj(edge_list):
        adj = [0] * N
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(N):
            neighbors = adj[u]
            n = neighbors
            while n:
                b = n & (-n)
                v = b.bit_length() - 1
                n -= b
                common = adj[u] & adj[v]
                if common & ~(1 << u):
                    return False
        return True
    
    def is_3_colorable(adj):
        color = [-1] * N
        order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            for c in range(3):
                ok = True
                neighbors = adj[v]
                n = neighbors
                while n:
                    b = n & (-n)
                    u = b.bit_length() - 1
                    n -= b
                    if color[u] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def count_edges(adj):
        total = 0
        for u in range(N):
            total += bin(adj[u]).count('1')
        return total // 2
        
    def edge_list_from_adj(adj):
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    adj = build_adj(groetzsch_edges)
    best_adj = adj[:]
    best_count = count_edges(adj)
    
    for v in range(11, 16):
        if time.time() > DEADLINE:
            break
        
        found = False
        # Iterate masks in decreasing order of bits
        # To be faster, we can generate masks by size
        # But simple range is okay for v <= 15
        
        # Optimization: Only consider masks that are independent
        # We can filter them out
        
        for mask in range((1 << v) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
                
            # Check independence
            is_ind = True
            m = mask
            while m:
                b1 = m & (-m)
                u1 = b1.bit_length() - 1
                m -= b1
                if adj[u1] & mask:
                    is_ind = False
                    break
            if not is_ind:
                continue
            
            new_adj = adj[:]
            new_adj[v] = mask
            for u in range(v):
                if mask & (1 << u):
                    new_adj[u] |= (1 << v)
            
            if not is_3_colorable(new_adj):
                adj = new_adj
                found = True
                break
        
        best_adj = adj[:]
        best_count = count_edges(adj)
        
    if is_triangle_free(best_adj) and not is_3_colorable(best_adj):
        return edge_list_from_adj(best_adj)
    else:
        return groetzsch_edges

