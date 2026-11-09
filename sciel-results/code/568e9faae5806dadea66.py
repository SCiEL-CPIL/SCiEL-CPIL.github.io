import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_edge_index(i, j, n):
        """Calculate index in flat list for pair (i,j) with i<j."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def find_violation(n, colors):
        """
        Returns (c, u, v, x, y, w, z) if a mono K2,2,2 of color c is found.
        """
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in [0, 1]:
            for u in range(n):
                for v in range(u + 1, n):
                    mask_nv = adj[c][u] & adj[c][v]
                    mask_nv &= ~(1 << u)
                    mask_nv &= ~(1 << v)
                    if mask_nv.bit_count() < 4:
                        continue
                    verts = [i for i in range(n) if (mask_nv >> i) & 1]
                    for i_idx in range(len(verts)):
                        x = verts[i_idx]
                        for j_idx in range(i_idx + 1, len(verts)):
                            y = verts[j_idx]
                            common = adj[c][x] & adj[c][y] & mask_nv
                            if common.bit_count() >= 2:
                                w = None
                                z = None
                                for k in range(n):
                                    if (common >> k) & 1:
                                        if w is None:
                                            w = k
                                        else:
                                            z = k
                                            break
                                return (c, u, v, x, y, w, z)
        return None

    # Target n=30
    n_target = 30
    
    # Generate Paley(29)
    n_paley = 29
    squares = {x * x % n_paley for x in range(1, n_paley)}
    colors_paley = [0 if (j - i) % n_paley in squares else 1 
                    for i in range(n_paley) for j in range(i + 1, n_paley)]
    
    def try_find_30():
        # Initialize colors for n=30
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Randomly initialize the last 29 edges
            new_edges = [random.randint(0, 1) for _ in range(n_paley)]
            colors = colors_paley + new_edges
            
            # Local Search
            for step in range(500):
                if time.time() > deadline - 2:
                    break
                    
                violation = find_violation(n_target, colors)
                
                if violation is None:
                    return colors
                
                c, u, v, x, y, w, z = violation
                
                edges = [
                    (u, x), (u, y), (v, x), (v, y),
                    (u, w), (u, z), (v, w), (v, z),
                    (x, w), (x, z), (y, w), (y, z)
                ]
                
                i, j = random.choice(edges)
                idx = get_edge_index(i, j, n_target)
                colors[idx] = 1 - colors[idx]
                
        return None

    # Try to find n=30
    colors_30 = try_find_30()
    if colors_30:
        if find_violation(n_target, colors_30) is None:
            return {'n': n_target, 'colors': colors_30}
        
    # Fallback to n=29 (Paley)
    if find_violation(n_paley, colors_paley) is None:
        return {'n': n_paley, 'colors': colors_paley}
        
    # Fallback to n=17
    n_fallback = 17
    squares_17 = {x * x % n_fallback for x in range(1, n_fallback)}
    colors_17 = [0 if (j - i) % n_fallback in squares_17 else 1 
                 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    return {'n': n_fallback, 'colors': colors_17}

# EVOLVE-BLOCK-END
