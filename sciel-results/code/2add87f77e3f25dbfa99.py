import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_n = 0
    best_colors = []
    
    def check_c4_addition(colors, n, u, v, c):
        """Check if adding edge (u,v) in color c creates a C4 in color c."""
        # Find all neighbors of u in color c
        # Find all neighbors of v in color c
        # Check if any neighbor a of u and neighbor b of v have edge (a,b) in color c
        adj_u = []
        adj_v = []
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if (min(u,w), max(u,w)) is in color c
            i, j = min(u, w), max(u, w)
            idx = i * n - i * (i + 1) // 2 + (j - i - 1)
            if colors[idx] == c:
                adj_u.append(w)
            i2, j2 = min(v, w), max(v, w)
            idx2 = i2 * n - i2 * (i2 + 1) // 2 + (j2 - i2 - 1)
            if colors[idx2] == c:
                adj_v.append(w)
        
        # Check if there exists a in adj_u, b in adj_v with (a,b) in color c
        for a in adj_u:
            for b in adj_v:
                if a == b:
                    continue
                i3, j3 = min(a, b), max(a, b)
                idx3 = i3 * n - i3 * (i3 + 1) // 2 + (j3 - i3 - 1)
                if colors[idx3] == c:
                    return True
        return False
    
    def get_idx(n, i, j):
        """Get index for edge (i,j) in the flat array."""
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def try_greedy(n, rng, time_limit):
        """Try to build a valid 5-coloring of K_n using greedy with backtracking."""
        total_edges = n * (n - 1) // 2
        colors = [-1] * total_edges
        
        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Shuffle edges for randomization
        rng.shuffle(edges)
        
        # Greedy assignment
        for (u, v) in edges:
            idx = get_idx(n, u, v)
            valid_colors = []
            for c in range(5):
                if not check_c4_addition(colors, n, u, v, c):
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Choose randomly among valid colors
            colors[idx] = rng.choice(valid_colors)
        
        # Final validation
        for c in range(5):
            # Build adjacency for color c
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = get_idx(n, i, j)
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
            # Check for C4: for each pair of vertices, check common neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if len(common) >= 2:
                        return None
        return colors
    
    # Try from large n down to small n
    for n in range(29, 10, -1):
        if time.time() > deadline:
            break
        
        # Multiple attempts for each n
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 1000 + n)
            result = try_greedy(n, rng, deadline - time.time())
            
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_colors = result
                break  # Found valid coloring for this n, move to smaller n
    
    # If we didn't find anything large, ensure we have at least the base case
    if best_n < 11:
        n = 11
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
