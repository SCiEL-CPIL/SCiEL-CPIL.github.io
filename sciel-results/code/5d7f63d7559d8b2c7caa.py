import time
import numpy as np

def construct():
    start_time = time.time()
    
    def search(n, time_limit):
        deadline = start_time + time_limit
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Vectorized NumPy adjacency matrix
        # adj_np[c][u][v] is 1 if (u,v) is color c
        adj_np = np.zeros((4, n, n), dtype=np.uint8)
        # Bitmasks for fast neighbor iteration in DFS
        adj_bits = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        if n < 4:
            return None
            
        def set_edge(u, v, c, idx):
            adj_np[c, u, v] = 1
            adj_np[c, v, u] = 1
            adj_bits[c][u] |= (1 << v)
            adj_bits[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj_np[c, u, v] = 0
            adj_np[c, v, u] = 0
            adj_bits[c][u] &= ~(1 << v)
            adj_bits[c][v] &= ~(1 << u)
            colors[idx] = -1

        # Symmetry breaking
        set_edge(0, 1, 0, 0)
        set_edge(0, 2, 1, 1)
        set_edge(0, 3, 2, 2)
        start_idx = 3
        
        def has_path_len_5(u, v, c):
            # DFS using bitmasks
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 5:
                    if node == v:
                        return True
                    continue
                
                neighbors = adj_bits[c][node]
                while neighbors:
                    lsb = neighbors & -neighbors
                    nxt = lsb.bit_length() - 1
                    neighbors ^= lsb
                    
                    if visited & lsb:
                        continue
                    
                    if depth == 4 and nxt == v:
                        return True
                        
                    stack.append((nxt, depth + 1, visited | lsb))
            return False

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            for c in range(4):
                if has_path_len_5(u, v, c):
                    continue
                
                set_edge(u, v, c, idx)
                
                if backtrack(idx + 1):
                    return True
                
                unset_edge(u, v, c, idx)
            
            return False

        if backtrack(start_idx):
            return {'n': n, 'colors': colors}
        return None

    # Try n=18
    result = search(18, 110)
    if result:
        return result
        
    # Try n=17
    result = search(17, 5)
    if result:
        return result
        
    # Fallback
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
