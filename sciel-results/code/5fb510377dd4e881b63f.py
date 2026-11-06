"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)

    # Build Conflict Graph as bitsets
    # adj[i] is an integer where bit j is 1 if i and j conflict
    adj = [0] * num_points

    def add_edge(u, v):
        if u == v:
            return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Type 1: w is apex. dist(u, w) == dist(v, w)
    # For each w, for each distance d2, all pairs in neighbors[w][d2] conflict.
    for w in range(num_points):
        xw, yw = points[w]
        buckets = {}
        for u in range(num_points):
            if u == w:
                continue
            xu, yu = points[u]
            dx = xw - xu
            dy = yw - yu
            d2 = dx*dx + dy*dy
            if d2 not in buckets:
                buckets[d2] = []
            buckets[d2].append(u)

        for lst in buckets.values():
            k = len(lst)
            if k < 2:
                continue
            # Add edges between all pairs in lst
            for a in range(k):
                u = lst[a]
                for b in range(a + 1, k):
                    v = lst[b]
                    add_edge(u, v)

    # Type 2: u is apex. dist(u, w) == dist(u, v)
    # For each u, for each w, let d2 = dist(u,w)^2.
    # All v in neighbors[u][d2] conflict with w.
    for u in range(num_points):
        xu, yu = points[u]
        buckets = {}
        for w in range(num_points):
            if w == u:
                continue
            xw, yw = points[w]
            dx = xu - xw
            dy = yu - yw
            d2 = dx*dx + dy*dy
            if d2 not in buckets:
                buckets[d2] = []
            buckets[d2].append(w)

        for lst in buckets.values():
            k = len(lst)
            if k < 2:
                continue
            # For each pair (w, v) in lst, add edge
            for a in range(k):
                w = lst[a]
                for b in range(a + 1, k):
                    v = lst[b]
                    add_edge(w, v)

    # Precompute static degrees and bit operations helpers
    static_deg = [bin(adj[i]).count('1') for i in range(num_points)]

    def popcount(x):
        return bin(x).count('1')

    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    # Greedy independent set heuristic for initial lower bound
    def greedy_iset(seed):
        random.seed(seed)
        remaining = (1 << num_points) - 1
        selected = []
        while remaining:
            # Pick node with min degree in remaining
            best_u = -1
            best_deg = float('inf')
            for u in iter_bits(remaining):
                deg = popcount(adj[u] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_u = u
            selected.append(best_u)
            remaining &= ~(1 << best_u)
            remaining &= ~adj[best_u]
        return selected

    # Branch and bound MIS solver with greedy coloring bound
    def branch_and_bound(initial_bound=0):
        # Order nodes by static degree ascending for better branching
        order = sorted(range(num_points), key=lambda i: static_deg[i])
        # Reindex to put low-degree nodes first
        pos = {node: idx for idx, node in enumerate(order)}
        rev = [0] * num_points
        for node, idx in pos.items():
            rev[idx] = node

        # Build reindexed adjacency
        adj2 = [0] * num_points
        for idx in range(num_points):
            node = rev[idx]
            m = adj[node]
            new_m = 0
            for j in iter_bits(m):
                new_m |= (1 << pos[j])
            adj2[idx] = new_m

        deg2 = [bin(adj2[i]).count('1') for i in range(num_points)]

        best_size = initial_bound
        best_set = set()

        # Greedy coloring bound: returns list of color classes (bitsets)
        # and number of colors used
        def greedy_color_bound(cand_mask):
            colors = []
            remaining = cand_mask
            while remaining:
                color = 0
                # Greedy: take vertices in order of increasing degree within remaining
                # For speed, we'll use a simple heuristic: pick vertex with min degree in remaining
                best_v = -1
                best_d = float('inf')
                for v in iter_bits(remaining):
                    d = popcount(adj2[v] & remaining)
                    if d < best_d:
                        best_d = d
                        best_v = v
                color |= (1 << best_v)
                remaining &= ~(1 << best_v)
                remaining &= ~adj2[best_v]
                colors.append(color)
            return colors

        # Recursive search
        def search(cand_mask, selected):
            nonlocal best_size, best_set
            if not cand_mask:
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = set(selected)
                return

            # Upper bound: current selected + greedy coloring bound on candidates
            colors = greedy_color_bound(cand_mask)
            if len(selected) + len(colors) <= best_size:
                return

            # Branch on a vertex: pick one with max degree in candidates for better pruning
            # Or min degree? Usually max degree for coloring-based bounds.
            # Let's pick vertex with max degree in candidates
            branch_v = -1
            max_d = -1
            for v in iter_bits(cand_mask):
                d = popcount(adj2[v] & cand_mask)
                if d > max_d:
                    max_d = d
                    branch_v = v

            # Branch 1: Include branch_v
            selected.append(branch_v)
            new_cand = cand_mask & ~(1 << branch_v) & ~adj2[branch_v]
            search(new_cand, selected)
            selected.pop()

            # Branch 2: Exclude branch_v
            new_cand = cand_mask & ~(1 << branch_v)
            search(new_cand, selected)

        # Initial greedy solution
        init_sol = greedy_iset(0)
        search((1 << num_points) - 1, [])

        if best_set:
            # Map back to original indices
            return [rev[i] for i in best_set]
        return init_sol

    # Try a few initial greedy solutions to set a good lower bound
    best_init = []
    for seed in range(5):
        sol = greedy_iset(seed)
        if len(sol) > len(best_init):
            best_init = sol

    # Run branch and bound with the best initial solution as lower bound
    final_sol = branch_and_bound(len(best_init))

    if not final_sol:
        final_sol = best_init

    return [points[i] for i in final_sol]

# EVOLVE-BLOCK-END
