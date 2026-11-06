import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Define groups of 4 vertices
    groups = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Internal color patterns for each group
    internal_colors = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Inter-group colors
    Z1 = 16
    Z2 = 17
    Z3 = 18
    
    # Initialize colors array
    colors = [0] * (N * (N - 1) // 2)
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Assign internal colors
    for g_idx, group in enumerate(groups):
        c0, c1, c2, c3 = internal_colors[g_idx]
        v0, v1, v2, v3 = group
        
        colors[get_idx(v0, v1)] = c0
        colors[get_idx(v2, v3)] = c0
        colors[get_idx(v0, v2)] = c1
        colors[get_idx(v1, v3)] = c1
        colors[get_idx(v0, v3)] = c2
        colors[get_idx(v1, v2)] = c3
        
    # Assign inter-group colors
    def assign_inter(g1_idx, g2_idx, color_val):
        for u in groups[g1_idx]:
            for v in groups[g2_idx]:
                colors[get_idx(u, v)] = color_val
                
    assign_inter(0, 1, Z1)
    assign_inter(2, 3, Z2)
    assign_inter(0, 2, Z3)
    assign_inter(0, 3, Z3)
    assign_inter(1, 2, Z3)
    assign_inter(1, 3, Z3)
    
    # Build adjacency structure for fast constraint checking
    # edge_color[u][v] = color of edge (u,v)
    edge_color = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            c = colors[get_idx(i, j)]
            edge_color[i][j] = c
            edge_color[j][i] = c
    
    # Precompute all 4-cycles for each edge
    # For edge (u,v), find all 4-cycles containing it
    # A 4-cycle through (u,v) is u-a-b-v-u where a,b are distinct from u,v
    # and a!=b
    # The three edges of the cycle besides (u,v) are: (u,a), (a,b), (b,v)
    # We need these 4 edges to not all have distinct colors
    
    # For each edge, store the list of 4-cycles it participates in
    # A 4-cycle is defined by 4 vertices {w,x,y,z} in some order
    # For edge (u,v), the other two vertices are a,b
    # The cycle is u-a-b-v (or u-b-a-v)
    
    # Precompute: for each pair (u,v), list of (a,b) pairs that form a 4-cycle
    # where the cycle edges are (u,v), (u,a), (a,b), (b,v)
    # We need: not all of {color(u,v), color(u,a), color(a,b), color(b,v)} are distinct
    
    edge_to_cycles = [[] for _ in range(N*(N-1)//2)]
    
    for u in range(N):
        for v in range(u+1, N):
            eu = get_idx(u, v)
            for a in range(N):
                if a == u or a == v:
                    continue
                for b in range(N):
                    if b == u or b == v or b == a:
                        continue
                    # 4-cycle: u-a-b-v
                    # Edges: (u,v), (u,a), (a,b), (b,v)
                    # Check that these form a valid 4-cycle (all 4 vertices distinct)
                    # They do by construction
                    edge_to_cycles[eu].append((a, b))
    
    def is_valid_coloring(colors):
        """Check if coloring has no rainbow 4-cycle."""
        # Rebuild edge_color
        ec = [[-1] * N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                c = colors[get_idx(i, j)]
                ec[i][j] = c
                ec[j][i] = c
        
        for i in range(N):
            for j in range(i+1, N):
                c1 = ec[i][j]
                for a in range(N):
                    if a == i or a == j:
                        continue
                    c2 = ec[i][a]
                    for b in range(N):
                        if b == i or b == j or b == a:
                            continue
                        c3 = ec[a][b]
                        c4 = ec[b][j]
                        if len({c1, c2, c3, c4}) == 4:
                            return False
        return True
    
    # Local search: try to change edge colors to introduce new colors
    # Strategy: for each edge, try changing its color to a new color
    # and check if the coloring remains valid
    
    used_colors = set(colors)
    
    # Get list of all edges
    all_edges = []
    for i in range(N):
        for j in range(i+1, N):
            all_edges.append((i, j))
    
    # Shuffle edges for randomization
    random.seed(42)
    
    # Try to introduce new colors
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Pick a random edge
        edge_idx = random.randint(0, len(all_edges)-1)
        u, v = all_edges[edge_idx]
        old_color = colors[get_idx(u, v)]
        
        # Try to change to a new color
        # Pick a random unused color
        unused = [c for c in range(120) if c not in used_colors]
        if not unused:
            break
        
        new_color = random.choice(unused)
        
        # Check if changing this edge to new_color keeps validity
        # We need to check all 4-cycles containing this edge
        # For 4-cycle u-a-b-v: colors are (u,v)=new_color, (u,a), (a,b), (b,v)
        # These 4 must not all be distinct
        
        valid = True
        for a in range(N):
            if a == u or a == v:
                continue
            ca = colors[get_idx(u, a)]
            for b in range(N):
                if b == u or b == v or b == a:
                    continue
                cb = colors[get_idx(a, b)]
                cd = colors[get_idx(b, v)]
                if len({new_color, ca, cb, cd}) == 4:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            colors[get_idx(u, v)] = new_color
            used_colors.add(new_color)
            # Also need to check 4-cycles where this edge is not the "first" edge
            # but we already checked all 4-cycles containing this edge
            # Wait, we need to be more careful. The 4-cycle constraint involves
            # all 4 edges. When we change edge (u,v), we need to check all 4-cycles
            # that contain (u,v). A 4-cycle containing (u,v) has the form:
            # u-x-y-v where x,y are other vertices.
            # The 4 edges are: (u,v), (u,x), (x,y), (y,v)
            # We checked all such cycles above.
            # But we also need to ensure we didn't break any cycle where
            # (u,v) appears in a different position. However, a 4-cycle is
            # just a set of 4 edges forming a cycle. If (u,v) is in the cycle,
            # then the other 3 edges connect the other two vertices to u and v.
            # So our check is complete.
        
        if time.time() > deadline:
            break
    
    # Final verification
    if not is_valid_coloring(colors):
        # Fallback to original coloring
        colors = [0] * (N * (N - 1) // 2)
        for g_idx, group in enumerate(groups):
            c0, c1, c2, c3 = internal_colors[g_idx]
            v0, v1, v2, v3 = group
            colors[get_idx(v0, v1)] = c0
            colors[get_idx(v2, v3)] = c0
            colors[get_idx(v0, v2)] = c1
            colors[get_idx(v1, v3)] = c1
            colors[get_idx(v0, v3)] = c2
            colors[get_idx(v1, v2)] = c3
        assign_inter(0, 1, Z1)
        assign_inter(2, 3, Z2)
        assign_inter(0, 2, Z3)
        assign_inter(0, 3, Z3)
        assign_inter(1, 2, Z3)
        assign_inter(1, 3, Z3)
    
    return colors
# EVOLVE-BLOCK-END
