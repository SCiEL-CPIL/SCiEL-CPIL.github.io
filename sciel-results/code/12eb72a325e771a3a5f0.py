    # Random Initialization
    MAX_C = 15
    colors = [[0]*N for _ in range(N)]
    color_counts = [0] * MAX_C
    for i in range(N):
        for j in range(i+1, N):
            c = random.randint(0, MAX_C - 1)
            colors[i][j] = c
            colors[j][i] = c
            color_counts[c] += 1
            
    # Function to get distinct colors in a K4
    def get_distinct_colors(k4_idx):
        edges = k4_edges[k4_idx]
        cols = set()
        for u, v in edges:
            cols.add(colors[u][v])
        return len(cols)
        
    # Initial bad count
    bad_count = 0
    bad_status = [False] * len(k4s)
    for idx in range(len(k4s)):
        if get_distinct_colors(idx) < 5:
            bad_count += 1
            bad_status[idx] = True
            
    # Count distinct colors used
    def count_distinct_colors():
        return sum(1 for c in color_counts if c > 0)
        
    current_colors_used = count_distinct_colors()
    
    # Simulated Annealing
    start_time = time.time()
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    W = 1000 # Weight for bad count
    
    while T > min_T:
        if time.time() - start_time > 110:
            break
            
        # Pick a random edge
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_c = colors[u][v]
        
        # Pick a random new color
        new_c = random.randint(0, MAX_C - 1)
        if new_c == old_c:
            continue
            
        # Calculate Delta
        affected = edge_to_k4s[(u, v)]
        delta_bad = 0
        
        for k4_idx in affected:
            edges = k4_edges[k4_idx]
            other_colors = set()
            for a, b in edges:
                if (a == u and b == v) or (a == v and b == u):
                    continue
                other_colors.add(colors[a][b])
                
            old_distinct = len(other_colors | {old_c})
            new_distinct = len(other_colors | {new_c})
            
            old_bad = old_distinct < 5
            new_bad = new_distinct < 5
            
            if old_bad and not new_bad:
                delta_bad -= 1
            elif not old_bad and new_bad:
                delta_bad += 1
                
        # Delta in colors used
        delta_colors = 0
        # If old_c count becomes 0, we lose a color.
        # If new_c count was 0, we gain a color.
        if color_counts[old_c] == 1:
            delta_colors -= 1
        if color_counts[new_c] == 0:
            delta_colors += 1
            
        delta_E = delta_bad * W + delta_colors
        
        # Acceptance
        if delta_E < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_E / T)
            
        if accept:
            colors[u][v] = new_c
            colors[v][u] = new_c
            color_counts[old_c] -= 1
            color_counts[new_c] += 1
            bad_count += delta_bad
            current_colors_used += delta_colors
            
            # Update bad_status
            for k4_idx in affected:
                bad_status[k4_idx] = get_distinct_colors(k4_idx) < 5
                
        T *= cooling
        
        if bad_count == 0 and current_colors_used <= 13:
             # We found a very good solution, break early?
             # 13 is the lower bound.
             break

    # Final Result
    res = []
    for i in range(N):
        for j in range(i+1, N):
            res.append(colors[i][j])
    return res
