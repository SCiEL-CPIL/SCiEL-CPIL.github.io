import time
import random

def count_c4s(n, colors):
    """
    Count monochromatic C4s.
    colors: list of length n*(n-1)//2, colors[i] is color of edge (i, j) where i<j.
    We need to map (i,j) to index.
    idx = i*n - i*(i+1)//2 + (j-i-1)
    """
    c4_count = 0
    # Build adjacency lists for each color
    # adj[c][u] = set of v
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    # For each color, check for C4
    for c in range(5):
        for u in range(n):
            # For each pair of neighbors of u in color c
            neighbors = list(adj[c][u])
            for k in range(len(neighbors)):
                v1 = neighbors[k]
                for l in range(k+1, len(neighbors)):
                    v2 = neighbors[l]
                    # Check if v1 and v2 have a common neighbor w (w != u)
                    # Intersection of adj[c][v1] and adj[c][v2]
                    common = adj[c][v1].intersection(adj[c][v2])
                    if u in common:
                        common.remove(u)
                    if len(common) >= 1:
                        # Found a C4: u-v1-w-v2-u
                        c4_count += 1
    return c4_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try n=26
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline:
            break
            
        # Random initialization
        m = n*(n-1)//2
        colors = [random.randint(0, 4) for _ in range(m)]
        
        # Local Search
        best_colors = colors[:]
        best_cost = count_c4s(n, best_colors)
        
        if best_cost == 0:
            return {'n': n, 'colors': best_colors}
            
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            # Pick random edge
            idx = random.randint(0, m-1)
            old_c = best_colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            best_colors[idx] = new_c
            new_cost = count_c4s(n, best_colors)
            
            if new_cost < best_cost:
                best_cost = new_cost
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors}
            else:
                best_colors[idx] = old_c
                
            # Occasionally accept worse to escape local minima
            if random.random() < 0.1 and new_cost < best_cost + 10:
                best_colors[idx] = new_c
                best_cost = new_cost
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors}
            else:
                best_colors[idx] = old_c
                
        if best_cost == 0:
            return {'n': n, 'colors': best_colors}
            
    # Fallback to seed n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
