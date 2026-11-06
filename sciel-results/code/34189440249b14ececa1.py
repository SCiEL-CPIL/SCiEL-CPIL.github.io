"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    n = 20
    colors = 20

    # We'll build a graph on vertices 0..19 with edges colored 0..19
    # such that every 4-cycle is rainbow.
    #
    # Key insight: A 4-cycle a-b-c-d-a has edges (a,b), (b,c), (c,d), (d,a).
    # For it to be rainbow, all four edge colors must be distinct.
    #
    # Strategy: Use a structured construction.
    # One approach: partition vertices into groups and use group-based coloring.
    #
    # Let's try a different structural approach:
    # Use a complete graph minus some edges, with a careful coloring scheme.
    #
    # Actually, let's think about what constraints exist.
    # If we have a 4-cycle, the four edges must have distinct colors.
    # This means: for any two edges that share a vertex and are part of a 4-cycle,
    # they cannot have the same color IF that 4-cycle exists.
    #
    # More precisely: if edges (u,v) and (w,x) are in some 4-cycle together with two other edges,
    # and (u,v) and (w,x) have the same color, that's a violation.
    #
    # Let's try a greedy/heuristic approach with backtracking limited by time.
    #
    # Alternative: Use a known construction.
    # Consider vertices as elements of Z_20. Color edge (i,j) by (i+j) mod 20.
    # Check: 4-cycle a,b,c,d. Edges: (a,b) color a+b, (b,c) color b+c, (c,d) color c+d, (d,a) color d+a.
    # For rainbow: a+b, b+c, c+d, d+a must be all distinct mod 20.
    # a+b = b+c => a=c (not allowed in simple cycle)
    # a+b = c+d => a-b-c-d pattern... a+b = c+d means a-c = d-b
    # This doesn't guarantee rainbow.
    #
    # Let's try: color edge (i,j) by (i-j) mod 20.
    # 4-cycle a,b,c,d: edges (a,b) color a-b, (b,c) color b-c, (c,d) color c-d, (d,a) color d-a = -(a-d).
    # Need a-b, b-c, c-d, d-a all distinct mod 20.
    # a-b = b-c => a+c = 2b
    # This can happen.
    #
    # Let's try a more systematic approach: use a 1-factorization or similar.
    #
    # Actually, let's try a completely different approach:
    # Start with the complete graph K_20 and remove edges to make it valid,
    # or build up from scratch.
    #
    # Let's try: partition vertices into 4 groups of 5.
    # Color edges within groups differently than between groups.
    #
    # Simpler idea: use a bipartite-like structure.
    # Split into two sets A and B of 10 vertices each.
    # Only use edges between A and B (complete bipartite K_{10,10}).
    # Color each edge (a_i, b_j) by (i + j) mod 20 where i is index in A, j in B.
    # Wait, we only have 20 colors and 100 edges, so many edges share colors.
    #
    # For a 4-cycle in K_{10,10}: a_i - b_j - a_k - b_l - a_i.
    # Edges: (a_i,b_j) color (i+j)%20, (b_j,a_k) color (k+j)%20, (a_k,b_l) color (k+l)%20, (b_l,a_i) color (i+l)%20.
    # Need these four to be distinct.
    # (i+j) = (k+j) => i=k (not a cycle)
    # (i+j) = (k+l) => i-j = l-k
    # This can fail.
    #
    # Let me try a different coloring for bipartite: color (a_i, b_j) by i*10 + j mod 20? No, that's only 100 values but we need mod 20.
    #
    # Let's just do a local search / simulated annealing on the edge set.
    #
    # Representation: For each pair (u,v) with u<v, decide if edge exists and its color.
    # But that's 190 edges * 20 colors = huge space.
    #
    # Better: Start with a valid small graph, try to add edges.
    #
    # Let's start with a star (valid, 19 edges) and try to add edges.
    #
    # Actually, let's try a completely different structural approach.
    #
    # Use vertices 0..19. Create edges based on a finite field or cyclic group structure.
    #
    # Let me try: for each vertex v, connect to v+1, v+2, v+4, v+8 mod 20 (powers of 2).
    # That's a 4-regular graph. Color edge (v, v+d) by d's index.
    # But we need to check 4-cycles.
    #
    # This is getting complicated. Let me just do a heuristic search.
    #
    # Approach: 
    # 1. Start with empty graph.
    # 2. Repeatedly try to add an edge (u,v) with some color c.
    # 3. Check if adding this edge creates any non-rainbow 4-cycle.
    # 4. If valid, add it. If not, try another color or skip.
    # 5. Use randomization and multiple restarts.
    #
    # Checking if adding edge (u,v) with color c creates a bad 4-cycle:
    # A 4-cycle containing (u,v) has form u-v-x-y-u where x,y are other vertices.
    # The other three edges are (v,x), (x,y), (y,u).
    # For this to be a 4-cycle, all four edges must exist and u,v,x,y distinct.
    # The colors must be distinct: c, color(v,x), color(x,y), color(y,u) all different.
    #
    # So when adding (u,v) with color c, for every pair (x,y) such that (v,x), (x,y), (y,u) all exist,
    # we need c != color(v,x), c != color(x,y), c != color(y,u).
    #
    # This is checkable in O(n^2) per edge addition.
    #
    # Let's implement this greedy approach with randomization.

    import random
    random.seed(42)

    def check_validity(adj, color_map):
        """Check if the current graph has all rainbow 4-cycles."""
        # adj[u] = set of neighbors
        # color_map[(u,v)] = color (u<v)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for x in adj[v]:
                    if x == u:
                        continue
                    for y in adj[x]:
                        if y == u or y == v:
                            continue
                        if y not in adj[u]:
                            continue
                        # Found 4-cycle u-v-x-y-u
                        c1 = color_map.get((min(u,v), max(u,v)))
                        c2 = color_map.get((min(v,x), max(v,x)))
                        c3 = color_map.get((min(x,y), max(x,y)))
                        c4 = color_map.get((min(y,u), max(y,u)))
                        if c1 is None or c2 is None or c3 is None or c4 is None:
                            continue
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True

    def can_add_edge(adj, color_map, u, v, c):
        """Check if adding edge (u,v) with color c creates a non-rainbow 4-cycle."""
        u, v = min(u,v), max(u,v)
        # For every x in adj[v] and y in adj[x] such that y in adj[u] and y != u,v and x != u,v
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v or y == x:
                    continue
                if y not in adj[u]:
                    continue
                # 4-cycle: u-v-x-y-u
                c1 = c  # (u,v)
                c2 = color_map.get((min(v,x), max(v,x)))
                c3 = color_map.get((min(x,y), max(x,y)))
                c4 = color_map.get((min(y,u), max(y,u)))
                if c2 is None or c3 is None or c4 is None:
                    continue
                if len({c1, c2, c3, c4}) < 4:
                    return False
        return True

    def try_build():
        adj = [set() for _ in range(n)]
        color_map = {}
        
        # Order of edges to try
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for attempt in range(50):
            random.shuffle(all_edges)
            for u, v in all_edges:
                if u in adj[v]:
                    continue
                # Try each color
                colors_to_try = list(range(colors))
                random.shuffle(colors_to_try)
                added = False
                for c in colors_to_try:
                    if can_add_edge(adj, color_map, u, v, c):
                        adj[u].add(v)
                        adj[v].add(u)
                        color_map[(u, v)] = c
                        added = True
                        break
                if not added:
                    continue
        
        # Count edges
        edge_count = sum(len(s) for s in adj) // 2
        return adj, color_map, edge_count

    best_adj = None
    best_color_map = None
    best_count = 0

    # Multiple restarts
    for restart in range(20):
        adj, color_map, count = try_build()
        if count > best_count:
            best_count = count
            best_adj = adj
            best_color_map = color_map
        
        if time.time() > deadline - 30:
            break

    # Try to improve with local search: try swapping colors
    if best_adj is not None and best_color_map is not None and time.time() < deadline - 10:
        # Try to add more edges or recolor
        adj = [set(s) for s in best_adj]
        color_map = dict(best_color_map)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
            random.shuffle(all_edges)
            for u, v in all_edges:
                colors_to_try = list(range(colors))
                random.shuffle(colors_to_try)
                for c in colors_to_try:
                    if can_add_edge(adj, color_map, u, v, c):
                        adj[u].add(v)
                        adj[v].add(u)
                        color_map[(u, v)] = c
                        improved = True
                        break

        edge_count = sum(len(s) for s in adj) // 2
        if edge_count > best_count:
            best_count = edge_count
            best_adj = adj
            best_color_map = color_map

    # Build result
    if best_adj is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, n)]

    result = []
    for (u, v), c in best_color_map.items():
        result.append([u, v, c])
    
    return result
# EVOLVE-BLOCK-END
