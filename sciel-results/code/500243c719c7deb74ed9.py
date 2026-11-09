"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, d, seed):
        from random import Random
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

    def popcount(x):
        return bin(x).count('1')

    def color_graph_bnb(edges, deadline):
        """
        Solve strong edge coloring via Minimum Clique Cover on conflict graph
        using branch-and-bound with clique number lower bound.
        """
        if not edges:
            return []

        m = len(edges)
        if m == 0:
            return []

        # Build adjacency list for original graph
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build conflict masks: two edges conflict if they share a vertex or are joined by an edge
        conflict_masks = [0] * m
        for i in range(m):
            u_i, v_i = edges[i]
            cov = {u_i, v_i}
            cov.update(adj.get(u_i, set()))
            cov.update(adj.get(v_i, set()))
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov or v_j in cov:
                    mask |= (1 << j)
            conflict_masks[i] = mask

        conflict_deg = [popcount(cm) for cm in conflict_masks]
        full_mask = (1 << m) - 1

        # Precompute: for each vertex in conflict graph, its non-neighbors (potential clique mates)
        # non_neighbor_mask[i] = bitmask of vertices that can be in a clique with i
        non_neighbor_mask = [0] * m
        for i in range(m):
            nm = full_mask & ~(conflict_masks[i] | (1 << i))
            non_neighbor_mask[i] = nm

        # Lower bound: clique number of a subgraph
        # Greedy clique lower bound
        def greedy_clique_lb(subset_mask):
            """Find a greedy clique in the induced subgraph, return its size as lower bound."""
            if not subset_mask:
                return 0
            # Greedy: repeatedly pick vertex with highest degree in remaining subgraph
            remaining = subset_mask
            clique_size = 0
            while remaining and time.time() < deadline:
                # Find vertex with max degree in remaining subgraph
                best_v = -1
                best_deg = -1
                for v in range(m):
                    if not (remaining >> v) & 1:
                        continue
                    deg = popcount(non_neighbor_mask[v] & remaining)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                clique_size += 1
                remaining &= non_neighbor_mask[best_v]
                remaining &= ~(1 << best_v)
            return clique_size

        # Upper bound: greedy coloring
        def greedy_coloring(order):
            color = [-1] * m
            color_used_by = []
            max_color = 0
            for i in order:
                cm = conflict_masks[i]
                taken = set()
                for c, mask in enumerate(color_used_by):
                    if cm & mask:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                if c >= len(color_used_by):
                    color_used_by.append(0)
                color_used_by[c] |= (1 << i)
                if c > max_color:
                    max_color = c
            return color, max_color

        # Initial upper bound via greedy
        order = sorted(range(m), key=lambda i: conflict_deg[i], reverse=True)
        init_color, ub = greedy_coloring(order)

        # Branch and bound
        best_color = init_color[:]
        best_ub = ub

        # State for B&B
        def bnb_solve(uncolored_mask, current_assignment, current_count, ub_limit):
            """
            Branch and bound to find clique cover.
            Returns (color_array, total_colors) or None if pruned.
            """
            nonlocal best_color, best_ub

            if not uncolored_mask:
                return current_assignment.copy(), current_count

            if current_count >= best_ub:
                return None

            # Lower bound: clique number of remaining subgraph
            lb = greedy_clique_lb(uncolored_mask)
            if current_count + lb >= best_ub:
                return None

            # Find a maximal clique (independent set in conflict graph) in uncolored
            # Greedy: pick vertex with fewest conflicts in remaining
            remaining = uncolored_mask
            clique = []
            while remaining and time.time() < deadline:
                # Pick vertex with min conflicts in remaining
                best_v = -1
                best_conf = m + 1
                for v in range(m):
                    if not (remaining >> v) & 1:
                        continue
                    conf = popcount(conflict_masks[v] & remaining)
                    if conf < best_conf:
                        best_conf = conf
                        best_v = v
                if best_v == -1:
                    break
                clique.append(best_v)
                # Remove best_v and its conflicts from remaining
                remaining &= ~(conflict_masks[best_v] | (1 << best_v))

            if not clique:
                # Fallback: color one vertex
                v = (uncolored_mask & -uncolored_mask).bit_length() - 1
                clique = [v]

            # Assign this clique a new color
            new_count = current_count + 1
            for v in clique:
                current_assignment[v] = new_count - 1

            # Recurse
            result = bnb_solve(uncolored_mask & ~(sum(1 << v for v in clique)), current_assignment, new_count, ub_limit)

            # Backtrack
            for v in clique:
                current_assignment[v] = -1

            if result is not None:
                return result

            # Try alternative: don't use this exact clique, try a different branching
            # This is where we could add more sophisticated branching, but for now
            # we rely on the greedy clique being good enough
            return None

        # Run B&B with time limit
        assignment = [-1] * m
        result = bnb_solve(full_mask, assignment, 0, best_ub)

        if result is not None:
            color_arr, total = result
            if total < best_ub:
                best_ub = total
                best_color = color_arr

        # If B&B didn't improve, use initial greedy
        return best_color

    # Fallback greedy coloring
    def color_graph_greedy(edges, deadline):
        if not edges:
            return []
        m = len(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        conflict_masks = [0] * m
        for i in range(m):
            u_i, v_i = edges[i]
            cov = {u_i, v_i}
            cov.update(adj.get(u_i, set()))
            cov.update(adj.get(v_i, set()))
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov or v_j in cov:
                    mask |= (1 << j)
            conflict_masks[i] = mask

        conflict_deg = [popcount(cm) for cm in conflict_masks]
        order = sorted(range(m), key=lambda i: conflict_deg[i], reverse=True)

        color = [-1] * m
        color_used_by = []
        for i in order:
            cm = conflict_masks[i]
            taken = set()
            for c, mask in enumerate(color_used_by):
                if cm & mask:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            color[i] = c
            if c >= len(color_used_by):
                color_used_by.append(0)
            color_used_by[c] |= (1 << i)
        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # Fallback to simple greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue

        edges = instance(*spec)
        colors = color_graph_bnb(edges, DEADLINE)
        out.append(colors)

    return out
# EVOLVE-BLOCK-END
