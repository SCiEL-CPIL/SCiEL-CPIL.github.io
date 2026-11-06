import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Lazebnik-Woldar construction on AG(2,5): 25 vertices
    # Vertices: (x, y) for x, y in F_5
    # Color edge between (x1,y1) and (x2,y2):
    #   if x1 != x2: slope = (y2-y1) * (x2-x1)^(-1) mod 5 -> color 0..4
    #   if x1 == x2: vertical -> need to assign a color
    # 
    # Problem: 6 directions, 5 colors. 
    # Solution: Use the fact that we can define a 5-coloring differently.
    # 
    # Alternative: Use vertices as F_5 x F_5, and color edge by:
    #   d = (x2-x1, y2-y1) normalized. There are 24 nonzero vectors, 
    #   grouped into 6 parallel classes of 4 vectors each (±v).
    #   We need 5 colors, so we merge two parallel classes.
    #
    # Actually, the known result: R_5(C4) > 25 is easy, R_5(C4) > 26 is the 
    # Lazebnik-Woldar result. Let me construct n=25 first.
    #
    # For n=25: use slope in F_5 for non-vertical, and for vertical edges,
    # use (y2-y1) mod 5 as color. But this might create C4s.
    #
    # Safer: Use the construction where color = (y2-y1) * (x2-x1)^(-1) mod 5
    # for x1!=x2, and for x1==x2, color = (y2-y1) mod 5.
    # Let's verify this is C4-free.
    
    n = 25
    colors = []
    for i in range(n):
        x1, y1 = divmod(i, 5)
        for j in range(i + 1, n):
            x2, y2 = divmod(j, 5)
            if x1 == x2:
                # Vertical edge
                c = (y2 - y1) % 5
            else:
                # Non-vertical: compute slope
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                # Inverse of dx mod 5
                inv_dx = pow(dx, -1, 5)
                c = (dy * inv_dx) % 5
            colors.append(c)
    
    # Verify no C4 in any color class
    def check_c4(colors, n):
        # For each color, check no two vertices have 2 common neighbors
        for color in range(5):
            # Build adjacency for this color
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == color:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            # Check for C4: two vertices with >= 2 common neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if len(common) >= 2:
                        return False, color, i, j, common
        return True, None, None, None, None
    
    valid, c, i, j, common = check_c4(colors, n)
    if not valid:
        # Fallback to known valid n=11 construction
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    # Try to extend to n=26 by adding vertex 25
    # Vertex 25 connects to vertices 0..24. Need to assign colors to avoid C4.
    # For each edge (25, v), choose a color that doesn't create a C4.
    # A C4 involving vertex 25: vertices 25, a, b, c where 25-a, a-b, b-c, c-25 form a cycle.
    # That means: color(25,a) = color(b,c) and color(25,c) = color(a,b), with a,b,c distinct.
    # More precisely: if we add vertex 25, a C4 in color k exists if there are vertices a,b,c
    # such that color(25,a)=k, color(25,c)=k, color(a,b)=k, color(b,c)=k, and a≠b≠c, a≠c.
    # Wait, C4 is 25-a-b-c-25, so edges (25,a),(a,b),(b,c),(c,25) all color k.
    # So: color(25,a)=k, color(a,b)=k, color(b,c)=k, color(25,c)=k.
    # This means: for each color k, the set of neighbors of 25 in color k, call N_k,
    # must not contain any two vertices a,c that are connected by a path of length 2 in color k
    # (i.e., a-b-c all in color k with b adjacent to both a and c in color k).
    # Equivalently: N_k must be an independent set in the color-k graph? No.
    # N_k must have no two vertices at distance 2 in the color-k graph.
    # I.e., for any a,c in N_k, there should be no b with color(a,b)=k and color(b,c)=k.
    
    # Build color-k adjacency structures
    color_adj = [set() for _ in range(5)]
    idx = 0
    for i in range(25):
        for j in range(i + 1, 25):
            c = colors[idx]
            color_adj[c].add((i, j))
            color_adj[c].add((j, i))
            idx += 1
    
    # Convert to sets for faster lookup
    color_neighbors = [[set() for _ in range(25)] for _ in range(5)]
    for c in range(5):
        for (u, v) in color_adj[c]:
            color_neighbors[c][u].add(v)
    
    # For each vertex v (0..24), determine which colors are "safe" for edge (25, v)
    # A color k is unsafe for edge (25,v) if there exists w such that:
    #   - color(25,w) is already assigned as k (or will be), AND
    #   - color(v,w) = k AND there's a path... 
    # This is complex. Let me use a greedy/backtracking approach.
    
    # Simpler: try all 5^25 assignments? No, too many.
    # Use greedy: for v=0,1,...,24, assign the smallest color that doesn't create a C4.
    
    # For a partial assignment of colors for edges (25, 0), ..., (25, v-1),
    # color k is valid for edge (25, v) if:
    #   For all w in {0,...,v-1} with color(25,w)=k:
    #     There is no u (any vertex 0..24) such that color(v,u)=k and color(u,w)=k
    #   AND for all u,v' in {0,...,v-1} with color(25,u)=k and color(25,v')=k:
    #     There is no w such that color(u,w)=k and color(w,v')=k (this is already checked by the above if we consider all pairs)
    
    # Actually, the C4 condition: adding edge (25,v) with color k creates a C4 iff
    # there exist distinct a,b in {0,...,v-1} such that color(25,a)=k, color(25,b)=k,
    # and there exists c (any vertex) with color(a,c)=k and color(c,b)=k, AND c≠25 (since 25 is new).
    # Wait, the C4 would be 25-a-c-b-25, so edges (25,a),(a,c),(c,b),(b,25) all color k.
    # So: color(25,a)=k, color(25,b)=k, color(a,c)=k, color(c,b)=k for some c≠25, a≠b, c distinct from a,b.
    
    # Precompute: for each color k and each pair (a,b), is there a c with color(a,c)=k and color(c,b)=k?
    # This is: are a and b at distance 2 in the color-k graph?
    
    dist2 = [[False] * 25 for _ in range(5)]  # dist2[k][a][b] = True if a,b at dist 2 in color k
    for k in range(5):
        for a in range(25):
            for b in range(25):
                if a == b:
                    continue
                # Check if there's c with (a,c) and (c,b) both in color k
                common = color_neighbors[k][a] & color_neighbors[k][b]
                if common:
                    dist2[k][a][b] = True
    
    # Now greedy assignment for vertex 25
    assign25 = [-1] * 25  # assign25[v] = color of edge (25, v)
    feasible = True
    for v in range(25):
        # Try each color
        found = False
        for k in range(5):
            # Check: for all w < v with assign25[w] == k, is dist2[k][v][w] False?
            ok = True
            for w in range(v):
                if assign25[w] == k and dist2[k][v][w]:
                    ok = False
                    break
            if ok:
                assign25[v] = k
                found = True
                break
        if not found:
            feasible = False
            break
    
    if feasible:
        # Build the full color list for n=26
        colors26 = list(colors)  # first 25*24/2 = 300 entries
        # Add edges (25, v) for v=0..24 in lexicographic order: (0,25),(1,25),...,(24,25)
        for v in range(25):
            colors26.append(assign25[v])
        return {'n': 26, 'colors': colors26}
    else:
        # Return n=25 construction
        return {'n': 25, 'colors': colors}
