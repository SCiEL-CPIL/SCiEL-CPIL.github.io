import time
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Build initial coloring using recursive splitting
    colors = [[0] * N for _ in range(N)]
    edge_list = []  # (i, j) for i < j
    
    for i in range(N):
        for j in range(i + 1, N):
            edge_list.append((i, j))
    
    # Recursive construction
    def build(vertices, next_color):
        n = len(vertices)
        if n <= 1:
            return next_color
        if n == 2:
            u, v = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            return next_color + 1
        if n == 3:
            u, v, w = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            colors[v][w] = next_color + 1
            colors[w][v] = next_color + 1
            colors[u][w] = next_color + 2
            colors[w][u] = next_color + 2
            return next_color + 3
        
        # Choose split point
        if n == 4:
            k = 1
        elif n == 5:
            k = 2
        elif n == 6:
            k = 3
        elif n == 7:
            k = 3
        elif n == 8:
            k = 4
        elif n == 9:
            k = 3
        elif n == 10:
            k = 4
        elif n == 11:
            k = 5
        elif n == 12:
            k = 6
        elif n == 13:
            k = 6
        elif n == 14:
            k = 7
        elif n == 15:
            k = 6
        elif n == 16:
            k = 7
        else:
            k = n // 2
        
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        cross_color = next_color
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        
        next_color += 1
        next_color = build(v1, next_color)
        if time.time() > deadline:
            return next_color
        next_color = build(v2, next_color)
        return next_color
    
    build(list(range(N)), 0)
    
    # Helper: check if recoloring edge (u,v) to new_color is valid
    def is_valid_recolor(u, v, new_color):
        # Check all C4s containing edge (u,v)
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                if x >= w:
                    continue
                # C4: u-w-x-v-u
                c1 = colors[u][w]
                c2 = colors[w][x]
                c3 = colors[x][v]
                c4 = new_color  # edge v-u
                if len(set([c1, c2, c3, c4])) > 3:
                    return False
        return True
    
    # Local search: try to add new colors
    current_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
    max_color_used = max(current_colors) if current_colors else 0
    
    # Precompute for each edge, the set of C4s it belongs to (for faster checking)
    # Actually, let's just do the check directly
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(edge_list)
        for (u, v) in edge_list:
            if time.time() > deadline:
                break
            old_color = colors[u][v]
            # Try to recolor to a new color (max_color_used + 1)
            new_color = max_color_used + 1
            if new_color > 119:
                continue
            if is_valid_recolor(u, v, new_color):
                colors[u][v] = new_color
                colors[v][u] = new_color
                max_color_used = new_color
                improved = True
    
    # Second pass: try to recolor edges to fill gaps in color usage
    used_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
    missing_colors = [c for c in range(120) if c not in used_colors]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(edge_list)
        for (u, v) in edge_list:
            if time.time() > deadline:
                break
            if not missing_colors:
                break
            old_color = colors[u][v]
            # Try each missing color
            for new_color in missing_colors[:]:
                if is_valid_recolor(u, v, new_color):
                    colors[u][v] = new_color
                    colors[v][u] = new_color
                    missing_colors.remove(new_color)
                    improved = True
                    break
    
    # Flatten to list of 120 integers
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(colors[i][j])
    
    return result
