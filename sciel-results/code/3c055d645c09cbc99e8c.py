"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i].add(j)
                    conflict[j].add(i)
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
        """
        Reduce the number of colours by repeatedly attacking the *most crowded*
        colour class (auxiliary-variable / epigraph view: push each class count
        below a target t). Uses single-edge moves and two-edge swaps where one
        edge moving to a lower colour frees a slot for the second edge.
        """
        n_edges = len(edges)
        current_max = max(colour)

        # Maintain per-colour counts for O(1) crowded-class selection.
        counts = {}
        for c in colour:
            counts[c] = counts.get(c, 0) + 1

        def counts_max():
            return max(counts.values()) if counts else 0

        def counts_top_color():
            # Return the colour with the largest count (tie -> highest colour,
            # which is the one we most want to dissolve).
            best_c = -1
            best_n = -1
            for c, n in counts.items():
                if n > best_n or (n == best_n and c > best_c):
                    best_n = n
                    best_c = c
            return best_c

        improved = True
        iterations = 0

        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1

            if current_max <= 0:
                break

            # Focus on the most crowded colour class (smooth surrogate for
            # "reduce the max"): even when the top colour is small, a large
            # class can often be dissolved to free a colour globally.
            focus = counts_top_color()
            if focus < 0:
                break

            focus_members = [i for i in range(n_edges) if colour[i] == focus]
            random.shuffle(focus_members)

            # --- Single-edge moves: try to move a focus edge to a lower colour ---
            for e_idx in focus_members:
                taken = set()
                for j in conflict[e_idx]:
                    taken.add(colour[j])
                new_c = 0
                while new_c in taken:
                    new_c += 1
                if new_c < focus:
                    # Move it.
                    counts[focus] -= 1
                    if counts[focus] == 0:
                        del counts[focus]
                    counts[new_c] = counts.get(new_c, 0) + 1
                    colour[e_idx] = new_c
                    # Update current_max.
                    if focus == current_max and counts.get(focus, 0) == 0:
                        while current_max >= 0 and counts.get(current_max, 0) == 0:
                            current_max -= 1
                    improved = True
                    break

            # --- Two-edge swaps: move one edge to a lower colour, freeing a
            #     colour that the second edge can then use. ---
            if not improved and current_max > 0 and time.time() < deadline - 5:
                focus = counts_top_color()
                if focus >= 0:
                    focus_members = [i for i in range(n_edges) if colour[i] == focus]
                    if len(focus_members) >= 2:
                        random.shuffle(focus_members)
                        sub = focus_members[:12]
                        for a in sub:
                            if improved:
                                break
                            # Colors edge a could move to (excluding focus).
                            avail_a = []
                            for c in range(focus):
                                # taken colors of a, ignoring b for now (b is
                                # also focus, so its colour is focus, not in
                                # range(focus)).
                                ok = True
                                for j in conflict[a]:
                                    if colour[j] == c:
                                        ok = False
                                        break
                                if ok:
                                    avail_a.append(c)
                            if not avail_a:
                                continue
                            for b in sub:
                                if a == b:
                                    continue
                                # Edge b's taken colours, ignoring a (a is focus,
                                # which we are trying to free).
                                taken_b = set()
                                for j in conflict[b]:
                                    if j != a:
                                        taken_b.add(colour[j])
                                # b needs a colour in range(focus) not in taken_b
                                # and different from the colour a takes.
                                for ca in avail_a[:4]:
                                    # b can take ca only if ca not in taken_b.
                                    # But a will occupy ca, so b cannot use ca.
                                    # b needs a different colour cb != ca, cb in
                                    # range(focus), cb not in taken_b.
                                    for cb in range(focus):
                                        if cb == ca:
                                            continue
                                        if cb in taken_b:
                                            continue
                                        # Assign a -> ca, b -> cb.
                                        counts[focus] -= 2
                                        if counts[focus] == 0:
                                            del counts[focus]
                                        counts[ca] = counts.get(ca, 0) + 1
                                        counts[cb] = counts.get(cb, 0) + 1
                                        colour[a] = ca
                                        colour[b] = cb
                                        if focus == current_max and counts.get(focus, 0) == 0:
                                            while current_max >= 0 and counts.get(current_max, 0) == 0:
                                                current_max -= 1
                                        improved = True
                                        break
                                    if improved:
                                        break
                                if improved:
                                    break
                        if improved:
                            pass

        # Recompute current_max defensively.
        if colour:
            current_max = max(colour)
        return colour, current_max + 1

    def try_multiple_orders(edges, adj, conflict, n_attempts=10):
        """Try multiple edge orderings and keep the best coloring."""
        n_edges = len(edges)
        constraint_deg = [len(conflict[i]) for i in range(n_edges)]

        best_colour = None
        best_max = float('inf')

        for attempt in range(n_attempts):
            if time.time() > deadline - 12:
                break

            if attempt == 0:
                order = sorted(range(n_edges), key=lambda i: -constraint_deg[i])
            elif attempt == 1:
                order = sorted(range(n_edges), key=lambda i: constraint_deg[i])
            elif attempt == 2:
                order = list(range(n_edges))
                random.shuffle(order)
            else:
                weights = [constraint_deg[i] + 1 for i in range(n_edges)]
                order = []
                remaining = list(range(n_edges))
                rem_weights = weights[:]
                for _ in range(n_edges):
                    if not remaining:
                        break
                    total_w = sum(rem_weights)
                    r = random.random() * total_w
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
        if time.time() > deadline - 25:
            break

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)

        # Try multiple orderings.
        colour, num_colors = try_multiple_orders(edges, adj, conflict, n_attempts=10)

        # Local improvement (main pass).
        if time.time() < deadline - 15:
            colour, num_colors = local_improve(edges, adj, colour, conflict, max_iter=50000)

        # High-precision polish pass: tighter budget, more iterations, to squeeze
        # out residual crowding the first pass left behind.
        if time.time() < deadline - 10:
            colour, num_colors = local_improve(edges, adj, colour, conflict, max_iter=30000)

        result = [colour[i] for i in range(n_edges)]
        out.append(result)

    # Fill remaining with simple greedy if time ran out.
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)

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
