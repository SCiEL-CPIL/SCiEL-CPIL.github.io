"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        """Precompute conflict matrix: conflict[i][j] = True if edges i and j conflict."""
        n_edges = len(edges)
        conflict = [bytearray(n_edges) for _ in range(n_edges)]
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Share a vertex
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i][j] = 1
                    conflict[j][i] = 1
                # Joined by an edge
                elif (ui in adj.get(uj, ()) or ui in adj.get(vj, ()) or
                      vi in adj.get(uj, ()) or vi in adj.get(vj, ())):
                    conflict[i][j] = 1
                    conflict[j][i] = 1
        return conflict

    def greedy_color(edges, adj, order, conflict):
        """Greedy coloring with given edge order, using precomputed conflict matrix."""
        colour = [0] * len(edges)
        for idx in order:
            taken = set()
            row = conflict[idx]
            for j in range(len(edges)):
                if row[j]:
                    taken.add(colour[j])
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c
        return colour

    def local_improve(edges, adj, colour, conflict, max_iter=20000):
        """Iteratively try to reduce the number of colors by reassigning edges."""
        n_edges = len(edges)
        current_max = max(colour)
        improved = True
        iterations = 0

        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1

            # Find edges with the highest color
            high_edges = [i for i in range(n_edges) if colour[i] == current_max]
            if not high_edges:
                current_max -= 1
                continue

            random.shuffle(high_edges)

            for e_idx in high_edges:
                # Try to reassign this edge to a lower color
                taken = set()
                row = conflict[e_idx]
                for j in range(n_edges):
                    if row[j]:
                        taken.add(colour[j])

                new_c = 0
                while new_c in taken:
                    new_c += 1

                if new_c < colour[e_idx]:
                    colour[e_idx] = new_c
                    # Check if max color decreased
                    if new_c < current_max and all(c != current_max for c in colour):
                        current_max -= 1
                    improved = True
                    break

            # Also try swapping: for each pair of edges with same high color,
            # see if swapping their colors helps
            if not improved and current_max > 0:
                high_edges2 = [i for i in range(n_edges) if colour[i] == current_max]
                if len(high_edges2) >= 2:
                    random.shuffle(high_edges2)
                    for a in high_edges2:
                        for b in high_edges2:
                            if a == b:
                                continue
                            # Try moving edge a to a lower color if we first move b
                            # Simple swap attempt
                            taken_a = set()
                            row_a = conflict[a]
                            for j in range(n_edges):
                                if row_a[j] and j != b:
                                    taken_a.add(colour[j])
                                elif row_a[j] and j == b:
                                    # b would have a different color after swap
                                    pass
                            # This is complex; skip for now
                            break

        return colour, current_max + 1

    def try_multiple_orders(edges, adj, conflict, n_attempts=5):
        """Try multiple edge orderings and keep the best coloring."""
        n_edges = len(edges)

        # Compute constraint degree for ordering
        constraint_deg = [0] * n_edges
        for i in range(n_edges):
            row = conflict[i]
            constraint_deg[i] = sum(row)

        best_colour = None
        best_max = float('inf')

        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break

            if attempt == 0:
                # Most constrained first
                order = sorted(range(n_edges), key=lambda i: -constraint_deg[i])
            elif attempt == 1:
                # Least constrained first
                order = sorted(range(n_edges), key=lambda i: constraint_deg[i])
            else:
                # Random order
                order = list(range(n_edges))
                random.shuffle(order)

            colour = greedy_color(edges, adj, order, conflict)
            max_c = max(colour) + 1

            if max_c < best_max:
                best_max = max_c
                best_colour = colour[:]

        return best_colour, best_max

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    for spec in specs:
        if time.time() > deadline - 20:
            # Not enough time, use simple greedy for remaining
            break

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)

        # Try multiple orderings
        colour, num_colors = try_multiple_orders(edges, adj, conflict, n_attempts=5)

        # Local improvement
        if time.time() < deadline - 15:
            colour, num_colors = local_improve(edges, adj, colour, conflict, max_iter=30000)

        result = [colour[i] for i in range(n_edges)]
        out.append(result)

    # Fill remaining with simple greedy if time ran out
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)

        # Simple greedy in natural order
        colour = [0] * n_edges
        for idx in range(n_edges):
            taken = set()
            row = conflict[idx]
            for j in range(n_edges):
                if row[j]:
                    taken.add(colour[j])
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c

        out.append(colour)

    return out
# EVOLVE-BLOCK-END
