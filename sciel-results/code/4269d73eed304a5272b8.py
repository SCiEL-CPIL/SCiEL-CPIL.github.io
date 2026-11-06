"""Strong edge colouring of three fixed bounded-degree graphs via MWIS branch-and-bound."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict
from random import Random


def construct():
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

    def build_conflict_graph(edges, adj):
        n = len(edges)
        conflict = [set() for _ in range(n)]
        for i in range(n):
            u1, v1 = edges[i]
            for j in range(i + 1, n):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def dsatur_coloring(conflict, num_vertices):
        colors = [-1] * num_vertices
        sat = [set() for _ in range(num_vertices)]
        degrees = [len(conflict[i]) for i in range(num_vertices)]
        uncolored = set(range(num_vertices))

        while uncolored:
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                d = degrees[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v

            used = sat[best_v]
            c = 0
            while c in used:
                c += 1
            colors[best_v] = c
            uncolored.remove(best_v)
            for u in conflict[best_v]:
                if u in uncolored:
                    sat[u].add(c)
        return colors

    def try_improve_with_bb(conflict, initial_colors, time_limit=10.0):
        import time
        start_time = time.time()
        n = len(conflict)
        if n == 0:
            return initial_colors

        # Get initial upper bound from DSATUR
        init_max_color = max(initial_colors)
        best_colors = initial_colors[:]
        best_k = init_max_color + 1

        # Build a simple ordering by degree (descending)
        order = sorted(range(n), key=lambda i: len(conflict[i]), reverse=True)
        # Map position in order to vertex
        pos_of = [0] * n
        for pos, v in enumerate(order):
            pos_of[v] = pos

        # For each color c, maintain which vertices can use it
        # We'll use a recursive backtracking with pruning

        # Precompute: for each vertex, the list of neighbors in conflict graph
        # We need to check if a vertex can be assigned a color

        # State: colors assigned so far, current max color used
        # We try to assign colors to vertices in 'order'

        # For efficiency, we track for each color which vertices are "blocked" (adjacent to a vertex with that color)
        # Actually, let's use a different approach:
        # For each vertex, maintain the set of colors that are forbidden

        forbidden = [set() for _ in range(n)]
        colors = [-1] * n

        # We process vertices in order
        # At each step, we try all available colors for the next vertex

        # Upper bound: if we've used max_color so far, and there are remaining vertices,
        # the minimum additional colors needed is at least 0, so total is at most max_color + 1
        # But we can be smarter: if all remaining vertices can fit in existing colors, no new color needed

        # Let's use a simple branch and bound:
        # - Process vertices in order
        # - For each vertex, try colors 0, 1, 2, ... in order
        # - Prune if current max color >= best_k

        def backtrack(idx, current_max):
            nonlocal best_colors, best_k, start_time
            if time.time() - start_time > time_limit:
                return
            if idx == n:
                if current_max + 1 < best_k:
                    best_k = current_max + 1
                    best_colors = colors[:]
                return

            v = order[idx]
            # Try colors 0, 1, 2, ...
            # A color c is available if c not in forbidden[v]
            # We can try colors in order 0, 1, 2, ...
            # Pruning: if current_max + 1 >= best_k and we're about to use a new color, skip
            # Actually, if current_max + 1 >= best_k, we can only use colors <= current_max
            # If no color <= current_max is available, we'd need a new color, which would make
            # total >= current_max + 2 > best_k (if current_max + 1 >= best_k)
            # So we can prune

            for c in range(0, best_k):
                if c in forbidden[v]:
                    continue
                # Assign color c to v
                colors[v] = c
                new_max = max(current_max, c)

                # Update forbidden for neighbors
                updated_neighbors = []
                for u in conflict[v]:
                    if colors[u] == -1:
                        forbidden[u].add(c)
                        updated_neighbors.append(u)

                if new_max + 1 < best_k or (new_max + 1 == best_k and idx + 1 < n):
                    backtrack(idx + 1, new_max)

                # Undo
                for u in updated_neighbors:
                    forbidden[u].discard(c)
                colors[v] = -1

                # If we found a solution with fewer colors, we can break
                if best_k <= current_max + 1:
                    break

        backtrack(0, -1)
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n_edges = len(edges)

        # Build original graph adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build conflict graph
        conflict = build_conflict_graph(edges, adj)

        # Get initial coloring with DSATUR
        colors = dsatur_coloring(conflict, n_edges)

        # Try to improve with branch and bound
        # Only try if the number of edges is small enough
        if n_edges <= 200:
            colors = try_improve_with_bb(conflict, colors, time_limit=15.0)

        out.append(colors)

    return out
# EVOLVE-BLOCK-END
