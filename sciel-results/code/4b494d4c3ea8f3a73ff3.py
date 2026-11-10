import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    # 2D list for colors
    color = [[0]*n for _ in range(n)]
    
    # Initialize with XOR
    for i in range(n):
        for j in range(i+1, n):
            val = i ^ j
            color[i][j] = val
            color[j][i] = val
            
    # Precompute K4s
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    k4s.append((a,b,c,d))
                    
    def get_colors(k4):
        a,b,c,d = k4
        return (color[a][b], color[a][c], color[a][d], color[b][c], color[b][d], color[c][d])
        
    def count_distinct(tup):
        return len(set(tup))
        
    # Initial cost
    cost = 0
    for k4 in k4s:
        cnt = count_distinct(get_colors(k4))
        if cnt < 5:
            cost += 5 - cnt
            
    # Precompute edge -> k4 indices
    edge_to_k4s = {}
    for idx, k4 in enumerate(k4s):
        a,b,c,d = k4
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        for e in edges:
            if e[0] > e[1]:
                e = (e[1], e[0])
            if e not in edge_to_k4s:
                edge_to_k4s[e] = []
            edge_to_k4s[e].append(idx)
            
    palette = list(range(1, 16))
    T = 5.0
    
    while time.time() < deadline:
        if cost == 0:
            break
            
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_c = color[i][j]
        new_c = random.choice(palette)
        if new_c == old_c:
            continue
            
        affected = edge_to_k4s.get((i,j), [])
        delta_cost = 0
        
        for idx in affected:
            k4 = k4s[idx]
            old_tup = get_colors(k4)
            old_cnt = count_distinct(old_tup)
            
            # Update
            color[i][j] = new_c
            color[j][i] = new_c
            new_tup = get_colors(k4)
            new_cnt = count_distinct(new_tup)
            
            # Revert
            color[i][j] = old_c
            color[j][i] = old_c
            
            delta_cost += max(0, 5 - new_cnt) - max(0, 5 - old_cnt)
            
        if delta_cost <= 0 or random.random() < pow(2.71828, -delta_cost / T):
            color[i][j] = new_c
            color[j][i] = new_c
            cost += delta_cost
            T *= 0.9995
        else:
            T *= 0.9995

    # Try to reduce colors
    # If valid, check if we can remove a color
    if cost == 0:
        for c in range(1, 16):
            # Find all edges with color c
            edges_to_change = []
            for i in range(n):
                for j in range(i+1, n):
                    if color[i][j] == c:
                        edges_to_change.append((i,j))
            
            if not edges_to_change:
                continue
                
            # Try to recolor to another color
            # Simple: try to recolor to color 1 if c != 1, or 2 if c == 1
            # But this might break validity.
            # We can use a local search specifically for this.
            # Given time, we skip this and return 15 colors.
            pass
            
    res = []
    for i in range(n):
        for j in range(i+1, n):
            res.append(color[i][j])
            
    return res
