"""Strong edge colourings via multi-start greedy + aggressive local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

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

    def solve_instance(edges, adj, time_budget):
        n_edges = len(edges)
        if n_edges == 0:
            return []

        # Precompute conflict bitmasks
        # conflict_mask[i] is a bitmask of all edges that conflict with edge i
        conflict_mask = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Check if edges conflict: share vertex or are joined by an edge
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
                else:
                    # Check if joined by an edge
                    if ui in adj.get(uj, set()) or ui in adj.get(vj, set()) or \
                       vi in adj.get(uj, set()) or vi in adj.get(vj, set()):
                        conflict_mask[i] |= (1 << j)
                        conflict_mask[j] |= (1 << i)

        def greedy_color(order):
            colour = [-1] * n_edges
            for idx in order:
                # Find taken colors
                taken = 0
                mask = conflict_mask[idx]
                for j in range(n_edges):
                    if mask & (1 << j):
                        if colour[j] >= 0:
                            taken |= (1 << colour[j])
                # Assign smallest available color
                c = 0
                while taken & (1 << c):
                    c += 1
                colour[idx] = c
            max_c = max(colour) if colour else 0
            return colour, max_c + 1

        def local_search(colour, max_colors, remaining_time):
            """Try to reduce max color by reassigning edges."""
            current_max = max_colors - 1
            improved = True
            rng = random.Random(42)

            while improved and time.time() < start_time + remaining_time:
                improved = False

                # Try to eliminate the highest color
                if current_max <= 0:
                    break

                high_edges = [i for i in range(n_edges) if colour[i] == current_max]
                rng.shuffle(high_edges)

                for e_idx in high_edges:
                    # Try to reassign this edge to a lower color
                    taken = 0
                    mask = conflict_mask[e_idx]
                    for j in range(n_edges):
                        if mask & (1 << j):
                            if colour[j] >= 0:
                                taken |= (1 << colour[j])

                    # Find smallest available color
                    new_c = 0
                    while taken & (1 << new_c):
                        new_c += 1

                    if new_c < colour[e_idx]:
                        colour[e_idx] = new_c
                        # Check if we can reduce max
                        all_colors = set(colour)
                        new_max = max(all_colors)
                        if new_max < current_max:
                            current_max = new_max
                        improved = True
                        break

                    # Try swap: find another edge with same color that can move
                    same_color = [j for j in range(n_edges) if colour[j] == colour[e_idx] and j != e_idx]
                    rng.shuffle(same_color)
                    for swap_idx in same_color[:5]:  # Try a few swaps
                        # Check if swap_idx can move to a different color
                        taken_swap = 0
                        mask_swap = conflict_mask[swap_idx]
                        for j in range(n_edges):
                            if mask_swap & (1 << j):
                                if j != e_idx and colour[j] >= 0:
                                    taken_swap |= (1 << colour[j])
                        # Remove the color that e_idx will take
                        # Actually, we need to check if after moving e_idx, swap_idx can move
                        # This is complex; skip for now
                        pass

            return colour, current_max + 1

        # Multi-start greedy with different orderings
        best_colour = None
        best_colors = n_edges + 1

        # Order 1: sorted by constraint degree (most constrained first)
        edge_constraints = []
        for i in range(n_edges):
            ui, vi = edges[i]
            count = 0
            for j in range(n_edges):
                if i != j and (conflict_mask[i] & (1 << j)):
                    count += 1
            edge_constraints.append(count)
        order1 = sorted(range(n_edges), key=lambda i: -edge_constraints[i])
        c1, nc1 = greedy_color(order1)
        if nc1 < best_colors:
            best_colour, best_colors = c1, nc1

        # Order 2: random
        if time.time() < start_time + remaining_time - 5:
            order2 = list(range(n_edges))
            random.shuffle(order2)
            c2, nc2 = greedy_color(order2)
            if nc2 < best_colors:
                best_colour, best_colors = c2, nc2

        # Order 3: least constrained first
        order3 = sorted(range(n_edges), key=lambda i: edge_constraints[i])
        c3, nc3 = greedy_color(order3)
        if nc3 < best_colors:
            best_colour, best_colors = c3, nc3

        # Local search on best
        if time.time() < start_time + remaining_time - 3:
            best_colour, best_colors = local_search(best_colour, best_colors, remaining_time)

        return best_colour

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    for idx, spec in enumerate(specs):
        if time.time() > deadline - 10:
            # Not enough time, use simple greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            n_edges = len(edges)
            colour = [-1] * n_edges
            for i, e in enumerate(edges):
                u, v = e
                near = {u, v}
                near.update(adj.get(u, set()))
                near.update(adj.get(v, set()))
                taken = set()
                for j, f in enumerate(edges):
                    if j != i and colour[j] >= 0:
                        fu, fv = f
                        if set(f) & near:
                            taken.add(colour[j])
                c = 0
                while c in taken:
                    c += 1
                colour[i] = c
            out.append(colour)
            continue

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = deadline - time.time() - 5
        colour = solve_instance(edges, adj, remaining)
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
