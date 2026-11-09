import time
import itertools

N = 16
DEADLINE = 85  # seconds, safety margin before 90s kill

# Precomputed optimal colors g(n) for n=1..16
G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]

# Precomputed best split k for n=4..16
SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    start_time = time.time()
    
    # Initialize the color matrix
    colors = [[0] * N for _ in range(N)]
    next_color = [0]
    
    def assign_colors(vertices):
        n = len(vertices)
        if n <= 1:
            return next_color[0]
        
        if n == 2:
            u, v = vertices
            c = next_color[0]
            colors[u][v] = c
            colors[v][u] = c
            next_color[0] += 1
            return next_color[0]
        
        if n == 3:
            u, v, w = vertices
            c1 = next_color[0]
            c2 = next_color[0] + 1
            c3 = next_color[0] + 2
            colors[u][v] = c1
            colors[v][u] = c1
            colors[v][w] = c2
            colors[w][v] = c2
            colors[u][w] = c3
            colors[w][u] = c3
            next_color[0] += 3
            return next_color[0]
        
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color[0] += 1
        
        assign_colors(v1)
        assign_colors(v2)
        
        return next_color[0]
    
    all_vertices = list(range(N))
    assign_colors(all_vertices)
    
    # Get current set of used colors
    used_colors = set()
    for i in range(N):
        for j in range(i+1, N):
            used_colors.add(colors[i][j])
    
    best_colors = [colors[i][j] for i in range(N) for j in range(i+1, N)]
    best_num_colors = len(used_colors)
    
    # Local search: try to increase color count by changing edges to new colors
    def is_valid_change(u, v, new_c):
        """Check if changing edge (u,v) to new_c creates no rainbow C4."""
        # Find all 4-cycles containing edge (u,v)
        # A 4-cycle with edge (u,v) has form u-v-w-x-u where w,x are distinct from u,v and each other
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                # Check cycle u-v-w-x-u
                c1 = colors[u][v] if colors[u][v] != new_c else new_c  # This is the edge being changed
                # Actually, let's just check the 4 edges of the cycle
                # Cycle: u-v, v-w, w-x, x-u
                e1 = new_c  # u-v
                e2 = colors[v][w]
                e3 = colors[w][x]
                e4 = colors[x][u]
                if len({e1, e2, e3, e4}) == 4:
                    return False
                # Also check the other direction: u-v-x-w-u is the same cycle reversed
                # But we should also check u-x-w-v-u which is a different 4-cycle
                # Actually for 4 vertices {u,v,w,x}, there are 3 distinct 4-cycles:
                # 1. u-v-w-x-u (edges: uv, vw, wx, xu)
                # 2. u-v-x-w-u (edges: uv, vx, xw, wu)  
                # 3. u-w-v-x-u (edges: uw, wv, vx, xu)
                # We already checked cycle 1. Now check cycle 2:
                e1 = new_c  # u-v
                e2 = colors[v][x]
                e3 = colors[x][w]
                e4 = colors[w][u]
                if len({e1, e2, e3, e4}) == 4:
                    return False
                # Cycle 3 doesn't contain edge uv, so skip
        return True
    
    # Try to improve
    edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for (u, v) in edges:
            if (time.time() - start_time) > DEADLINE:
                break
            old_c = colors[u][v]
            # Try new colors not currently used
            for new_c in range(120):
                if new_c == old_c or new_c in used_colors:
                    continue
                if is_valid_change(u, v, new_c):
                    # Accept the change
                    colors[u][v] = new_c
                    colors[v][u] = new_c
                    used_colors.add(new_c)
                    best_num_colors = len(used_colors)
                    improved = True
                    break  # Move to next edge
    
    # Build result
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(colors[i][j])
    
    return result
