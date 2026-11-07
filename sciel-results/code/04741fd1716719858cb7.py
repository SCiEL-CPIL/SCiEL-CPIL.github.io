import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Current edges set
    current_edges = set(base_edges)
    
    # Initialize distance matrix
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            D[i][j] = abs(i - j)
            
    added_edges = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    while True:
        if time.time() > deadline:
            break
            
        # Check diameter
        max_dist = 0
        for i in range(n):
            for j in range(n):
                if D[i][j] > max_dist:
                    max_dist = D[i][j]
                    
        if max_dist <= 4:
            break
            
        best_edge = None
        best_count = -1
        best_new_D = None
        
        # Generate candidates
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in current_edges:
                    candidates.append((u, v))
                    
        for u, v in candidates:
            # Check triangle freedom
            # A triangle is formed if there exists w such that (u,w) and (w,v) are in current_edges
            is_safe = True
            for w in range(n):
                if w != u and w != v:
                    if (u, w) in current_edges and (w, v) in current_edges:
                        is_safe = False
                        break
            if not is_safe:
                continue
                
            # Calculate new distances
            new_D = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    d1 = D[i][j]
                    d2 = D[i][u] + 1 + D[v][j]
                    d3 = D[i][v] + 1 + D[u][j]
                    new_D[i][j] = min(d1, d2, d3)
                    
            # Count fixed pairs
            count_fixed = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if D[i][j] > 4 and new_D[i][j] <= 4:
                        count_fixed += 1
                        
            if count_fixed > best_count:
                best_count = count_fixed
                best_edge = (u, v)
                best_new_D = new_D
                
        if best_edge is None:
            break
            
        # Add the edge
        added_edges.append(list(best_edge))
        current_edges.add(best_edge)
        current_edges.add((best_edge[1], best_edge[0]))
        
        # Update D
        D = best_new_D
        
    return added_edges
# EVOLVE-BLOCK-END
