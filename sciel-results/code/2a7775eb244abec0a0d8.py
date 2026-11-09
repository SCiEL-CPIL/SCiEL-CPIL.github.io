import time
import random
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill
    
    # Build initial coloring using recursive splitting
    colors = [[0] * N for _ in range(N)]
    edge_list = []  # (i, j) for i < j
    
    for i in range(N):
        for j in range(i + 1, N):
            edge_list.append((i, j))
    
    # Precompute all 4-cycles (sets of 4 vertices)
    # For each edge (u,v), precompute the list of (w,x) pairs such that {u,v,w,x} forms a 4-cycle
    # A 4-cycle on {u,v,w,x} has the cycle u-w-x-v-u, so edges are (u,w),(w,x),(x,v),(v,u)
    # We need to check that not all 4 edges have distinct colors
    
    # For faster checking: for each edge (u,v), store all pairs (w,x) with w<x, w!=u,v, x!=u,v, x!=w
    # such that the 4-cycle u-w-x-v-u exists (which it always does in K_n for any 4 distinct vertices)
    
    # Actually, for any 4 distinct vertices {a,b,c,d}, there are 3 distinct 4-cycles:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # Each cycle uses 4 edges. We need to ensure none of these 3 cycles has 4 distinct colors.
    
    # Let's precompute for each edge (u,v), the list of other edges that form a 4-cycle with (u,v)
    # Edge (u,v) is in a 4-cycle with edges (u,w),(w,x),(x,v) for distinct w,x not in {u,v}
    # The 4-cycle is u-w-x-v-u
    
    # For each edge (u,v), collect all (w,x) pairs where w,x not in {u,v}, w<x
    # The 4-cycle edges are: (u,w), (w,x), (x,v), (v,u)
    
    edge_to_cycles = {}  # (u,v) -> list of (edge1, edge2, edge3) where these 3 edges + (u,v) form a 4-cycle
    for i in range(N):
        for j in range(i+1, N):
            cycles = []
            for w in range(N):
                if w == i or w == j:
                    continue
                for x in range(N):
                    if x == i or x == j or x == w:
                        continue
                    if x >= w:
                        continue
                    # 4-cycle: i-w-x-j-i
                    # Edges: (i,w), (w,x), (x,j), (j,i)
                    # Sort edges for consistency
                    e1 = (min(i,w), max(i,w))
                    e2 = (min(w,x), max(w,x))
                    e3 = (min(x,j), max(x,j))
                    cycles.append((e1, e2, e3))
            edge_to_cycles[(i,j)] = cycles
    
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
        key = (min(u,v), max(u,v))
        for (e1, e2, e3) in edge_to_cycles[key]:
            c1 = colors[e1[0]][e1[1]]
            c2 = colors[e2[0]][e2[1]]
            c3 = colors[e3[0]][e3[1]]
            if len(set([c1, c2, c3, new_color])) > 3:
                return False
        return True
    
    # Local search: try to add new colors
    current_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
    max_color_used = max(current_colors) if current_colors else 0
    
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
    
    # Third pass: try swaps to free up colors for new ones
    # For each pair of edges, try swapping their colors to see if it enables a new color
    # This is more expensive, so do it with a time limit
    
    used_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
    max_color_used = max(used_colors) if used_colors else 0
    missing_colors = [c for c in range(120) if c not in used_colors]
    
    if missing_colors and time.time() < deadline:
        # Try to find an edge that can be recolored to a missing color after a swap
        # For simplicity, try random swaps
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            random.shuffle(edge_list)
            for idx1 in range(len(edge_list)):
                if time.time() > deadline - 2:
                    break
                if not missing_colors:
                    break
                u1, v1 = edge_list[idx1]
                for idx2 in range(idx1+1, len(edge_list)):
                    if time.time() > deadline - 2:
                        break
                    u2, v2 = edge_list[idx2]
                    c1 = colors[u1][v1]
                    c2 = colors[u2][v2]
                    if c1 == c2:
                        continue
                    # Try swapping
                    colors[u1][v1] = c2
                    colors[v1][u1] = c2
                    colors[u2][v2] = c1
                    colors[v2][u2] = c1
                    
                    # Check if any edge can now be recolored to a missing color
                    for (u, v) in edge_list:
                        for mc in missing_colors[:]:
                            if is_valid_recolor(u, v, mc):
                                colors[u][v] = mc
                                colors[v][u] = mc
                                missing_colors.remove(mc)
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        # Undo swap
                        colors[u1][v1] = c1
                        colors[v1][u1] = c1
                        colors[u2][v2] = c2
                        colors[v2][u2] = c2
    
    # Flatten to list of 120 integers
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(colors[i][j])
    
    return result
