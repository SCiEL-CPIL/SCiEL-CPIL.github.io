"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 20
    n_colors = 20
    
    # We'll build a graph and try to maximize edges with valid 4-cycle rainbow coloring.
    # Strategy: Start with a structured graph, then try to add edges.
    
    # Key insight: If we use a proper edge-coloring (each color class is a matching),
    # then in any 4-cycle, only opposite edges can share a color.
    # So we need: for any 4-cycle, opposite edges have different colors.
    
    # Let's try: c(u,v) = (u + v) % n for a complete graph, then remove violating edges.
    # Or better: use a known construction.
    
    # Approach: Use a bipartite graph with structured coloring, then add edges.
    
    # Let's try a greedy approach:
    # 1. Build edge list for K_20
    # 2. Assign colors using a formula
    # 3. Check all 4-cycles, remove edges that violate
    
    # Actually, let's try a different approach:
    # Use the fact that if each color appears on a set of edges forming a matching,
    # and no two edges of the same color are "opposite" in any 4-cycle, we're good.
    
    # Simpler: try random colorings of K_20, check validity, keep best valid subgraph.
    
    # First, let's establish a baseline: a star (19 edges) is always valid.
    best_edges = [[0, v, 0] for v in range(1, n)]
    best_count = len(best_edges)
    
    # Helper: check if a set of colored edges is valid (all 4-cycles are rainbow)
    def is_valid(edge_list):
        # Build adjacency with colors
        # edge_list: list of [u, v, c]
        adj = {}  # (u,v) -> color, assume u < v
        for u, v, c in edge_list:
            if u > v:
                u, v = v, u
            adj[(u, v)] = c
        
        # Find all 4-cycles and check they're rainbow
        # A 4-cycle: a-b-c-d-a where all four edges exist and a,b,c,d distinct
        # To avoid duplicates, we can iterate over pairs of non-adjacent vertices
        # that have two common neighbors.
        
        # Build neighbor lists
        neighbors = {i: {} for i in range(n)}  # neighbors[i][j] = color
        for (u, v), c in adj.items():
            neighbors[u][v] = c
            neighbors[v][u] = c
        
        # For each pair of vertices (a, c) that are NOT adjacent,
        # find common neighbors b, d such that a-b-c-d-a forms a 4-cycle
        # Actually, a 4-cycle is a-b-c-d-a. Vertices a,b,c,d all distinct.
        # Edges: (a,b), (b,c), (c,d), (d,a)
        
        # Enumerate: for each ordered pair (a,c) with a < c, find all pairs (b,d)
        # such that b and d are both neighbors of both a and c, b != d, b != a, b != c, d != a, d != c
        # and b != d. Then check the 4-cycle a-b-c-d-a.
        
        # Wait, that's not quite right. Let me think again.
        # 4-cycle: a-b-c-d-a. The edges are (a,b), (b,c), (c,d), (d,a).
        # So a and c are opposite, b and d are opposite.
        # a and c need not be adjacent (in fact they aren't in the cycle).
        # b and c are adjacent, a and b are adjacent, c and d are adjacent, d and a are adjacent.
        
        # So: for each pair (a, c) with a < c:
        #   common_neighbors = {x : (a,x) in adj and (c,x) in adj}
        #   for each pair (b, d) in common_neighbors with b < d, b != a, b != c, d != a, d != c:
        #     check 4-cycle a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
        #     all four must have distinct colors
        
        for a in range(n):
            for c in range(a+1, n):
                # Find common neighbors of a and c
                common = set(neighbors[a].keys()) & set(neighbors[c].keys())
                common.discard(a)
                common.discard(c)
                common_list = list(common)
                for i in range(len(common_list)):
                    for j in range(i+1, len(common_list)):
                        b = common_list[i]
                        d = common_list[j]
                        # 4-cycle: a-b-c-d-a
                        # Edges: (a,b), (b,c), (c,d), (d,a)
                        e1 = (min(a,b), max(a,b))
                        e2 = (min(b,c), max(b,c))
                        e3 = (min(c,d), max(c,d))
                        e4 = (min(d,a), max(d,a))
                        if e1 in adj and e2 in adj and e3 in adj and e4 in adj:
                            colors = [adj[e1], adj[e2], adj[e3], adj[e4]]
                            if len(set(colors)) != 4:
                                return False
        return True
    
    # Try structured constructions
    # Construction 1: Complete bipartite K_{10,10} with coloring c(u,v) = (u + v) % 20
    # Parts: A = {0..9}, B = {10..19}
    # Check if this is valid
    def try_bipartite():
        edges = []
        for a in range(10):
            for b in range(10, 20):
                c = (a + b) % n_colors
                edges.append([a, b, c])
        if is_valid(edges):
            return edges
        return None
    
    bip = try_bipartite()
    if bip is not None and len(bip) > best_count:
        best_edges = bip
        best_count = len(bip)
    
    # Construction 2: Try K_{10,10} with c(u,v) = (u * 2 + v) % 20
    def try_bipartite2():
        edges = []
        for a in range(10):
            for b in range(10, 20):
                c = (a * 2 + b) % n_colors
                edges.append([a, b, c])
        if is_valid(edges):
            return edges
        return None
    
    bip2 = try_bipartite2()
    if bip2 is not None and len(bip2) > best_count:
        best_edges = bip2
        best_count = len(bip2)
    
    # Construction 3: Try various linear colorings on K_{10,10}
    for alpha in range(20):
        for beta in range(20):
            if alpha == 0 and beta == 0:
                continue
            edges = []
            valid = True
            for a in range(10):
                for b in range(10, 20):
                    c = (alpha * a + beta * b) % n_colors
                    edges.append([a, b, c])
            if len(edges) > best_count and is_valid(edges):
                best_edges = edges
                best_count = len(edges)
                break
        if time.time() > deadline - 5:
            break
    
    # Try to add edges to the best graph found
    # Greedy: try adding each missing edge with a color that doesn't violate
    def try_add_edges(edge_list, max_time):
        edges = [list(e) for e in edge_list]
        adj = {}
        for u, v, c in edges:
            key = (min(u,v), max(u,v))
            adj[key] = c
        
        neighbors = {i: {} for i in range(n)}
        for (u, v), c in adj.items():
            neighbors[u][v] = c
            neighbors[v][u] = c
        
        # Try adding each missing edge
        all_possible = []
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) not in adj:
                    all_possible.append((u,v))
        
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > max_time:
                break
            # Try each color
            for c in range(n_colors):
                # Check if adding (u,v,c) creates any invalid 4-cycle
                # New 4-cycles involving edge (u,v):
                # u-v-x-y-u where x,y are common neighbors of u and v (other than each other)
                # Or u-x-v-y-u ... wait, let me think.
                # A 4-cycle containing edge (u,v): u-v-x-y-u or u-x-y-v-u etc.
                # Actually: the 4-cycle has edge (u,v), so the cycle is u-v-?-?-u
                # Let's say u-v-w-z-u. Then edges are (u,v), (v,w), (w,z), (z,u).
                # w must be neighbor of v, z must be neighbor of u, and (w,z) must be an edge.
                # Also need u,v,w,z all distinct.
                
                valid = True
                # 4-cycles of form u-v-w-z-u
                for w in neighbors[v]:
                    if w == u or w == v:
                        continue
                    for z in neighbors[u]:
                        if z == u or z == v or z == w:
                            continue
                        wz_key = (min(w,z), max(w,z))
                        if wz_key in adj:
                            colors = [c, adj[(min(u,v),max(u,v))] if False else None]  # placeholder
                            # Colors: (u,v)->c, (v,w)->adj[(min(v,w),max(v,w))], (w,z)->adj[wz_key], (z,u)->adj[(min(z,u),max(z,u))]
                            cw = adj[(min(v,w), max(v,w))]
                            cz = adj[wz_key]
                            czu = adj[(min(z,u), max(z,u))]
                            if len({c, cw, cz, czu}) != 4:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    edges.append([u, v, c])
                    adj[(u,v)] = c
                    neighbors[u][v] = c
                    neighbors[v][u] = c
                    break
        
        return edges
    
    # Try to improve
    if time.time() < deadline - 10:
        improved = try_add_edges(best_edges, deadline - 10)
        if len(improved) > best_count:
            best_edges = improved
            best_count = len(improved)
    
    return best_edges
# EVOLVE-BLOCK-END
