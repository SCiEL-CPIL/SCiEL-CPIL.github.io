import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Random edge ordering
        random.shuffle(all_pairs)
        
        # adj[u][v] = color or -1
        adj = [[-1]*n for _ in range(n)]
        # neighbors[u] = list of v such that adj[u][v] != -1
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            
            # Try to find a valid color
            # Shuffle colors to avoid bias
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                valid = True
                # Check all x such that (v, x) exists
                # x cannot be u
                for x in neighbors[v]:
                    if x == u:
                        continue
                    cx = adj[v][x]
                    
                    # Check all y such that (x, y) exists and (y, u) exists
                    # y cannot be u, v, x
                    # Intersection of neighbors[x] and neighbors[u]
                    # We can just iterate neighbors[x] and check if y in neighbors[u]
                    for y in neighbors[x]:
                        if y == u or y == v or y == x:
                            continue
                        if y not in neighbors[u]:
                            continue
                        
                        xy = adj[x][y]
                        yu = adj[y][u]
                        
                        # Colors: c, cx, xy, yu
                        # Must be distinct
                        if c == cx or c == xy or c == yu or cx == xy or cx == yu or xy == yu:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
            # Optional: If we hit a very high count, we might want to stop early?
            # But we want to maximize.
            
    return best_edges
