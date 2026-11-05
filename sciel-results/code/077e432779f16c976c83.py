import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    vertices = list(range(1, n + 1))
    
    invalid = set()
    for v in vertices:
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r == s:
            invalid.add(v)
            
    valid_vertices = [v for v in vertices if v not in invalid]
    idx_map = {v: i for i, v in enumerate(valid_vertices)}
    n_valid = len(valid_vertices)
    
    max_sum = 2 * n
    squares = set()
    k = 2
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    adj_masks = [0] * n_valid
    
    for i in range(n_valid):
        u = valid_vertices[i]
        mask = 0
        for j in range(n_valid):
            if i == j:
                continue
            v = valid_vertices[j]
            if (u + v) not in squares:
                mask |= (1 << j)
        adj_masks[i] = mask
        
    best_clique = []
    best_size = 0
    
    # Greedy initial solution
    def greedy_is():
        remaining = set(range(n_valid))
        clique = []
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (adj_masks[v] >> u) & 1)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            clique.append(valid_vertices[best_v])
            remaining.discard(best_v)
            # Remove neighbors
            to_remove = set()
            for u in remaining:
                if (adj_masks[best_v] >> u) & 1:
                    to_remove.add(u)
            remaining -= to_remove
        return clique

    init_clique = greedy_is()
    best_size = len(init_clique)
    best_clique = init_clique
    
    def expand(clique, candidates, depth):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if depth > best_size:
                best_size = depth
                best_clique = clique.copy()
            return
            
        verts = [i for i in range(n_valid) if (candidates >> i) & 1]
        
        if depth + len(verts) <= best_size:
            return
            
        # Simple coloring bound (approximate)
        # We just use len(verts) for now.
        
        # Branch on vertex with max degree
        max_deg = -1
        best_v = -1
        for v in verts:
            deg = (adj_masks[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        if best_v == -1:
            return

        # Include
        new_clique = clique + [valid_vertices[best_v]]
        new_candidates = candidates & adj_masks[best_v]
        expand(new_clique, new_candidates, depth + 1)
        
        # Exclude
        new_candidates = candidates & ~(1 << best_v)
        expand(clique, new_candidates, depth)
        
    initial_candidates = (1 << n_valid) - 1
    expand([], initial_candidates, 0)
    
    return best_clique
