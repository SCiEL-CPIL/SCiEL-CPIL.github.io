"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    from random import Random
    from itertools import combinations

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

    def solve_one(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency for vertices
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Build conflict graph: two edges conflict if they share a vertex or are joined by an edge
        # Conflict if:
        #   - they share a vertex, OR
        #   - there is an edge between a vertex of e1 and a vertex of e2
        # i.e., the two edges are not in the same induced matching
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # For each edge, compute the set of edges that conflict with it
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if there's an edge between a vertex of e1 and a vertex of e2
                # i.e., is (u1,u2) or (u1,v2) or (v1,u2) or (v1,v2) an edge?
                # We can check via adjacency
                if u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or \
                   u2 in adj.get(v1, set()) or v2 in adj.get(v1, set()):
                    conflict[i].add(j)
                    conflict[j].add(i)

        # Now we want to find the minimum number of induced matchings that cover all edges.
        # This is equivalent to coloring the conflict graph: each color class is an independent set
        # in the conflict graph, which corresponds to an induced matching in the original graph.
        
        # Use a greedy + local search approach for better results, but let's try a smarter greedy first.
        # Actually, let's use a greedy coloring with a good ordering, then try to improve.
        
        # Greedy coloring of the conflict graph
        # Order edges by descending degree in conflict graph (most constrained first)
        order = sorted(range(m), key=lambda i: -len(conflict[i]))
        
        color_of = [-1] * m
        used_colors = 0
        
        for idx in order:
            # Find available colors
            forbidden = set()
            for c_idx in conflict[idx]:
                if color_of[c_idx] != -1:
                    forbidden.add(color_of[c_idx])
            # Assign smallest available color
            c = 0
            while c in forbidden:
                c += 1
            color_of[idx] = c
            if c >= used_colors:
                used_colors = c + 1
        
        # Try to improve with local search: swap colors to reduce total
        # Simple approach: try to recolor each edge to a lower color if possible
        improved = True
        while improved:
            improved = False
            for idx in range(m):
                current = color_of[idx]
                if current == 0:
                    continue
                forbidden = set()
                for c_idx in conflict[idx]:
                    if color_of[c_idx] != -1:
                        forbidden.add(color_of[c_idx])
                # Try to find a lower color not in forbidden
                for c in range(current):
                    if c not in forbidden:
                        color_of[idx] = c
                        improved = True
                        break
        
        # Count used colors
        max_color = max(color_of) if m > 0 else 0
        return [color_of[i] for i in range(m)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        out.append(solve_one(*spec))
    return out
# EVOLVE-BLOCK-END
