"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    from collections import defaultdict

    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def conflict_set(edge, adj):
        """Vertices that conflict with this edge (endpoints + neighbors)."""
        u, v = edge
        s = {u, v}
        s.update(adj[u])
        s.update(adj[v])
        return s

    def greedy_color(edges, adj, order=None):
        """Greedy coloring with given edge order. Returns dict edge->color."""
        if order is None:
            order = list(edges)
        colour = {}
        # For each edge, precompute which colors are blocked
        # We track for each vertex, which colors are used by edges touching that vertex
        # An edge (u,v) conflicts with color c if some edge of color c touches u or v
        # i.e., there exists edge (a,b) with colour (a,b)=c and {a,b} & {u,v,adj[u],adj[v]} != empty
        # Equivalently: edge (a,b) conflicts with edge (u,v) if they share a vertex OR
        # are joined by an edge, i.e., there's a path of length 1 or 2 between them.
        # More precisely: two edges conflict if any endpoint of one is adjacent to
        # any endpoint of the other (including sharing a vertex).
        
        # Efficient approach: for each color, maintain the set of vertices "covered"
        # by edges of that color. An edge (u,v) can use color c iff no vertex in
        # conflict_set((u,v), adj) is covered by color c.
        
        color_vertices = defaultdict(set)  # color -> set of vertices covered by edges of that color
        
        for e in order:
            u, v = e
            near = conflict_set(e, adj)
            # Find smallest color not used
            c = 0
            while c in color_vertices and (color_vertices[c] & near):
                c += 1
            colour[e] = c
            color_vertices[c].add(u)
            color_vertices[c].add(v)
        
        return colour

    def count_colors(colour):
        if not colour:
            return 0
        return max(colour.values()) + 1

    def try_recolor(edge, new_color, colour, adj):
        """Try to recolor edge to new_color. Returns True if successful."""
        near = conflict_set(edge, adj)
        # Check if new_color is available
        for other_e, c in colour.items():
            if c == new_color:
                ou, ov = other_e
                if (ou in near) or (ov in near):
                    return False
        colour[edge] = new_color
        return True

    def local_improve(colour, edges, adj, max_iterations=2000):
        """Try to reduce number of colors via recoloring."""
        current_max = count_colors(colour)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            for e in edges:
                c = colour[e]
                # Try to recolor to a lower color
                for new_c in range(c):
                    if try_recolor(e, new_c, colour, adj):
                        improved = True
                        break
                # Also try to see if we can remove the highest color
            # Check if we reduced max
            new_max = count_colors(colour)
            if new_max < current_max:
                current_max = new_max
            if not improved:
                break
        return colour

    def dsatur_order(edges, adj):
        """Order edges by saturation: prioritize edges whose endpoints have many
        distinct colors in their neighborhoods (like DSATUR for vertices)."""
        # For strong edge coloring, an edge (u,v) is "saturated" by the set of colors
        # appearing on edges that conflict with it.
        # We process edges one by one; at each step, pick the edge with highest
        # "saturation" = number of distinct colors among conflicting edges.
        
        remaining = set(edges)
        colour = {}
        order = []
        
        # Track for each vertex, the set of colors on edges incident to that vertex
        # and on edges adjacent to edges incident to that vertex
        # Actually, for edge (u,v), conflicting edges are those sharing u or v,
        # or those with an endpoint adjacent to u or v.
        
        # Simpler: just use greedy with a static ordering based on degree
        # or use a dynamic approach
        
        # Dynamic DSATUR-like:
        # For each remaining edge, compute saturation = number of distinct colors
        # on edges that conflict with it.
        
        # This is expensive to recompute each time, so let's use a simpler heuristic:
        # Order by (degree(u) + degree(v)) descending, breaking ties by edge index.
        
        deg = {v: len(adj[v]) for v in adj}
        order = sorted(edges, key=lambda e: (deg.get(e[0], 0) + deg.get(e[1], 0), e), reverse=True)
        return order

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = build_adj(edges)
        
        # Try multiple orderings and keep the best
        best_colour = None
        best_count = float('inf')
        
        # Ordering 1: by degree sum descending
        deg = {v: len(adj[v]) for v in adj}
        order1 = sorted(edges, key=lambda e: (deg.get(e[0], 0) + deg.get(e[1], 0), e), reverse=True)
        colour1 = greedy_color(edges, adj, order1)
        local_improve(colour1, edges, adj)
        c1 = count_colors(colour1)
        if c1 < best_count:
            best_count = c1
            best_colour = dict(colour1)
        
        if time.time() > deadline:
            out.append([best_colour[e] for e in edges])
            continue
        
        # Ordering 2: random
        rng = Random(42)
        order2 = list(edges)
        rng.shuffle(order2)
        colour2 = greedy_color(edges, adj, order2)
        local_improve(colour2, edges, adj)
        c2 = count_colors(colour2)
        if c2 < best_count:
            best_count = c2
            best_colour = dict(colour2)
        
        if time.time() > deadline:
            out.append([best_colour[e] for e in edges])
            continue
        
        # Ordering 3: by degree sum ascending (least constrained first)
        order3 = sorted(edges, key=lambda e: (deg.get(e[0], 0) + deg.get(e[1], 0), e))
        colour3 = greedy_color(edges, adj, order3)
        local_improve(colour3, edges, adj)
        c3 = count_colors(colour3)
        if c3 < best_count:
            best_count = c3
            best_colour = dict(colour3)
        
        out.append([best_colour[e] for e in edges])
    
    return out
# EVOLVE-BLOCK-END
