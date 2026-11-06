import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145 seconds to be safe
    
    best_n = 9
    # Seed coloring for n=9
    n_seed = 9
    best_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]

    for n in range(17, 8, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid coloring for this n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [None] * len(edges)
        
        # Bitmasks for each color: adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        # Path masks: paths[k][c][u] is a bitmask of vertices reachable from u by a path of length k in color c
        paths = [[[0] * n for _ in range(4)] for _ in range(6)]
        
        # Initialize paths
        for c in range(4):
            for i in range(n):
                paths[0][c][i] = 1 << i
            for k in range(1, 6):
                for i in range(n):
                    mask = 0
                    neighbors = adj[c][i]
                    while neighbors:
                        lsb = neighbors & -neighbors
                        j = lsb.bit_length() - 1
                        mask |= paths[k-1][c][j]
                        neighbors ^= lsb
                    paths[k][c][i] = mask

        def recompute_paths(c):
            for k in range(1, 6):
                for i in range(n):
                    mask = 0
                    neighbors = adj[c][i]
                    while neighbors:
                        lsb = neighbors & -neighbors
                        j = lsb.bit_length() - 1
                        mask |= paths[k-1][c][j]
                        neighbors ^= lsb
                    paths[k][c][i] = mask

        def add_edge(c, u, v):
            if (paths[5][c][u] >> v) & 1:
                return False
            if (paths[5][c][v] >> u) & 1:
                return False
            
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            recompute_paths(c)
            return True
        
        def remove_edge(c, u, v):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            recompute_paths(c)

        def solve(idx):
            if time.time() > deadline - 2.0:
                return False
            
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            # Heuristic: try colors in a random order
            colors_order = [0, 1, 2, 3]
            random.shuffle(colors_order)
            
            for c in colors_order:
                if add_edge(c, u, v):
                    colors[idx] = c
                    if solve(idx + 1):
                        return True
                    colors[idx] = None
                    remove_edge(c, u, v)
                    
            return False

        # Run search with multiple restarts if needed
        found = False
        restarts = 0
        while not found and time.time() < deadline - 5.0:
            if solve(0):
                found = True
                best_n = n
                best_colors = colors[:]
                break
            restarts += 1
            # Reset state for next restart
            for c in range(4):
                for i in range(n):
                    adj[c][i] = 0
            # Recompute initial paths (all zero adj)
            for c in range(4):
                for i in range(n):
                    paths[0][c][i] = 1 << i
                for k in range(1, 6):
                    for i in range(n):
                        paths[k][c][i] = 0

    return {'n': best_n, 'colors': best_colors}
