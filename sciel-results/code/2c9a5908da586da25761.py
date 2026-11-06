import numpy as np
import random
import time

def find_first_c6(adj_list, n):
    """Find one C6 in the graph defined by adj_list. Returns tuple or None."""
    for i in range(n):
        for j in adj_list[i]:
            for k in adj_list[j]:
                if k == i: continue
                for l in adj_list[k]:
                    if l in (i, j): continue
                    for m in adj_list[l]:
                        if m in (i, j, k): continue
                        for p in adj_list[m]:
                            if p in (i, j, k, l): continue
                            if p != j and (i in adj_list[p]): # adj_list is symmetric, so p-i is edge
                                return (i, j, k, l, m, p)
    return None

def construct():
    target_n = 17
    random.seed(42)
    start_time = time.time()
    
    best_result = None
    
    # Try to find for n=17
    for n_try in [17, 16]:
        if time.time() - start_time > 100:
            break
            
        found = False
        # Increase attempts for smaller n if 17 fails?
        max_attempts = 100 if n_try == 17 else 50
        
        for attempt in range(max_attempts):
            if time.time() - start_time > 110:
                break
                
            # Random init
            m = n_try * (n_try - 1) // 2
            colors = [random.randint(0, 3) for _ in range(m)]
            
            # Precompute index map for speed
            # idx_map[(i,j)] = index
            idx_map = {}
            idx = 0
            for i in range(n_try):
                for j in range(i+1, n_try):
                    idx_map[(i, j)] = idx
                    idx += 1
            
            # Build adjacency lists
            def build_adj_lists(colors, n):
                adj = [set() for _ in range(4)]
                for i in range(n):
                    for j in range(i+1, n):
                        c = colors[idx_map[(i, j)]]
                        adj[c].add((i, j))
                        adj[c].add((j, i))
                # Convert to lists for faster iteration? Or keep sets?
                # Sets allow O(1) lookup for "is edge".
                # But we need to iterate neighbors.
                # Let's build adjacency lists (list of neighbors) for each color
                adj_lists = [[[] for _ in range(n)] for _ in range(4)]
                for c in range(4):
                    for (u, v) in adj[c]:
                        adj_lists[c][u].append(v)
                return adj_lists

            adj_lists = build_adj_lists(colors, n_try)
            
            # Local Search
            for iteration in range(2000):
                if time.time() - start_time > 110:
                    break
                    
                # Check for violations
                violation = None
                for c in range(4):
                    c6 = find_first_c6(adj_lists[c], n_try)
                    if c6:
                        violation = (c, c6)
                        break
                
                if violation is None:
                    best_result = {'n': n_try, 'colors': colors[:]}
                    found = True
                    break
                
                c, c6 = violation
                # Recolor one edge
                edges = [(c6[0], c6[1]), (c6[1], c6[2]), (c6[2], c6[3]), (c6[3], c6[4]), (c6[4], c6[5]), (c6[5], c6[0])]
                u, v = random.choice(edges)
                
                idx = idx_map[(min(u,v), max(u,v))]
                
                # Try a random new color
                new_color = random.randint(0, 3)
                while new_color == c:
                    new_color = random.randint(0, 3)
                
                # Update structures
                # Remove from c
                adj[c].discard((u, v))
                adj[c].discard((v, u))
                # Add to new_color
                adj[new_color].add((u, v))
                adj[new_color].add((v, u))
                
                # Update lists
                adj_lists[c][u].remove(v)
                adj_lists[c][v].remove(u)
                adj_lists[new_color][u].append(v)
                adj_lists[new_color][v].append(u)
                
                # Update colors array
                colors[idx] = new_color
                
            if found:
                break
        
        if found:
            return best_result
            
    # Fallback
    return {'n': 9, 'colors': [0]*81}
