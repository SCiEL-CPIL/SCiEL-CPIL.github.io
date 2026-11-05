"""Strong edge colourings of three fixed bounded-degree graphs."""
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

    def build_conflict(edges, adj):
        """Precompute conflict sets: conflict[i] is a set of edges conflicting with i."""
        n_edges = len(edges)
        conflict = [set() for _ in range(n_edges)]
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Share a vertex
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i].add(j)
                    conflict[j].add(i)
                # Joined by an edge
                elif (ui in adj.get(uj, set()) or ui in adj.get(vj, set()) or
                      vi in adj.get(uj, set()) or vi in adj.get(vj, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def greedy_color(edges, adj, order, conflict):
        """Greedy coloring with given edge order, using precomputed conflict sets."""
        colour = [0] * len(edges)
        for idx in order:
            taken = set()
            for j in conflict[idx]:
                taken.add(colour[j])
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c
        return colour

    def local_improve(edges, adj, colour, conflict, max_iter=50000):
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
                for j in conflict[e_idx]:
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

            # Try 2-opt style swaps: move two edges from high color to lower colors
            if not improved and current_max > 0 and time.time() < deadline - 5:
                high_edges2 = [i for i in range(n_edges) if colour[i] == current_max]
                if len(high_edges2) >= 2:
                    random.shuffle(high_edges2)
                    for a in high_edges2[:10]:
                        for b in high_edges2[:10]:
                            if a == b:
                                continue
                            # Try moving edge a to color c_a and edge b to color c_b
                            # where c_a != c_b and both are < current_max
                            # Check if a can go to some color c_a not used by its neighbors (except b)
                            # and b can go to some color c_b not used by its neighbors (except a)
                            taken_a = set()
                            for j in conflict[a]:
                                if j != b:
                                    taken_a.add(colour[j])
                            taken_b = set()
                            for j in conflict[b]:
                                if j != a:
                                    taken_b.add(colour[j])
                            
                            # Find available colors for a and b
                            avail_a = []
                            avail_b = []
                            for c in range(current_max):
                                if c not in taken_a:
                                    avail_a.append(c)
                                if c not in taken_b:
                                    avail_b.append(c)
                            
                            if avail_a and avail_b:
                                # Try to assign a to avail_a[0] and b to avail_b[0]
                                c_a = avail_a[0]
                                c_b = avail_b[0]
                                if c_a != c_b:
                                    colour[a] = c_a
                                    colour[b] = c_b
                                    improved = True
                                    break
                                else:
                                    # Try other combinations
                                    for ca in avail_a[:3]:
                                        for cb in avail_b[:3]:
                                            if ca != cb:
                                                colour[a] = ca
                                                colour[b] = cb
                                                improved = True
                                                break
                                        if improved:
                                            break
                            if improved:
                                break
                        if improved:
                            break
                # Check if max color decreased
                if improved:
                    if all(c != current_max for c in colour):
                        current_max -= 1

        return colour, current_max + 1

    def try_multiple_orders(edges, adj, conflict, n_attempts=10):
        """Try multiple edge orderings and keep the best coloring."""
        n_edges = len(edges)

        # Compute constraint degree for ordering
        constraint_deg = [len(conflict[i]) for i in range(n_edges)]

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
            elif attempt == 2:
                # Random order
                order = list(range(n_edges))
                random.shuffle(order)
            else:
                # Weighted random: prioritize higher constraint degree
                weights = [constraint_deg[i] + 1 for i in range(n_edges)]
                total_w = sum(weights)
                order = []
                remaining = list(range(n_edges))
                rem_weights = weights[:]
                for _ in range(n_edges):
                    if not remaining:
                        break
                    r = random.random() * sum(rem_weights)
                    cum = 0
                    chosen = None
                    for idx, w in enumerate(rem_weights):
                        cum += w
                        if r <= cum:
                            chosen = idx
                            break
                    if chosen is None:
                        chosen = len(rem_weights) - 1
                    order.append(remaining[chosen])
                    rem_weights.pop(chosen)
                    remaining.pop(chosen)

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
        colour, num_colors = try_multiple_orders(edges, adj, conflict, n_attempts=10)

        # Local improvement
        if time.time() < deadline - 15:
            colour, num_colors = local_improve(edges, adj, colour, conflict, max_iter=50000)

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
            for j in conflict[idx]:
                taken.add(colour[j])
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c

        out.append(colour)

    return out
# EVOLVE-BLOCK-END
