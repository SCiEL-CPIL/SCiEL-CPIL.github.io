import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def try_construct(n, time_limit=10.0):
        """
        Greedy constructive algorithm with backtracking.
        """
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Shuffle edges for different search paths
        random.shuffle(edges)
        
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        colors = [-1] * num_edges
        
        def get_idx(u, v):
            if u > v: u, v = v, u
            return u * n - u * (u + 1) // 2 + v - u - 1

        def is_valid(u, v, c):
            mask_v = adj[c][v]
            mask_u = adj[c][u]
            temp = mask_v
            while temp:
                lsb = temp & -temp
                w = lsb.bit_length() - 1
                if (mask_u & adj[c][w]) != 0:
                    return False
                temp ^= lsb
            return True

        def add_edge(u, v, c, idx):
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)

        def remove_edge(u, v, c, idx):
            colors[idx] = -1
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)

        def dfs(edge_idx):
            if time.time() > start_time + time_limit:
                return False
            
            if edge_idx == num_edges:
                return True
            
            u, v = edges[edge_idx]
            
            # Try colors 0-4
            for c in range(5):
                if is_valid(u, v, c):
                    add_edge(u, v, c, edge_idx)
                    if dfs(edge_idx + 1):
                        return True
                    remove_edge(u, v, c, edge_idx)
            
            return False

        if dfs(0):
            return colors
        
        return None

    # Try n=26 with a few restarts
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        result = try_construct(26, time_limit=20.0)
        if result is not None:
            return {'n': 26, 'colors': result}

    # Try n=25
    if time.time() < deadline - 5:
        result = try_construct(25, time_limit=10.0)
        if result is not None:
            return {'n': 25, 'colors': result}
            
    # Fallback
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
