"""Rewritten with a flow-based structural approach."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def max_flow(s, t, adj, cap):
        """Edmonds-Karp max flow. adj: dict node->list of neighbor nodes, cap: dict (u,v)->cap."""
        from collections import deque
        def bfs():
            parent = {s: None}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj.get(u, []):
                    if v not in parent and cap.get((u, v), 0) > 0:
                        parent[v] = u
                        if v == t:
                            return parent
                        q.append(v)
            return None
        total = 0
        while True:
            parent = bfs()
            if parent is None:
                break
            # find bottleneck
            path = []
            v = t
            while v != s:
                u = parent[v]
                path.append((u, v))
                v = u
            bottleneck = min(cap[(u, v)] for (u, v) in path)
            for (u, v) in path:
                cap[(u, v)] -= bottleneck
                cap[(v, u)] = cap.get((v, u), 0) + bottleneck
            total += bottleneck
        return total

    def dsatur_coloring(vertices, adj):
        if not vertices:
            return {}
        colors = {}
        sat_deg = {v: 0 for v in vertices}
        color_sets = {v: set() for v in vertices}
        uncolored = set(vertices)
        while uncolored:
            best = max(uncolored, key=lambda x: (sat_deg[x], len(adj[x] & uncolored)))
            c = 0
            while c in color_sets[best]:
                c += 1
            colors[best] = c
            color_sets[best].add(c)
            uncolored.discard(best)
            for w in adj[best]:
                if w in uncolored and c not in color_sets[w]:
                    color_sets[w].add(c)
                    sat_deg[w] = len(color_sets[w])
        return colors

    def solve_instance(n, k, p, seed):
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        I_set = set(I)

        rng = random.Random(seed)
        N = {v: set() for v in I}
        for v in I:
            for u in C:
                if rng.random() < p:
                    N[v].add(u)

        # All C-C edges
        all_c_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_c_edges.add((u, v))

        # Each C-I edge must be covered by exactly one clique
        # Model: for each v in I, partition N[v] into groups S_1, ..., S_m
        # Clique {v} ∪ S_i covers C-I edges (u,v) for u in S_i, and C-C edges within S_i
        #
        # Strategy: Use a flow to decide which "slots" each v uses.
        # But simpler: for each v, choose a partition of N[v] into cliques in C.
        # The C-C edges within those cliques get "used" (covered).
        # Residual C-C edges must be covered by C-only cliques.
        #
        # To minimize total cliques:
        # - For each v, fewer groups = fewer cliques involving v, but each group
        #   must be a clique in C (i.e., all pairs in group must be C-C edges, which
        #   they always are since C is a clique). Wait - C is a clique, so ANY subset
        #   of C is a clique! So for each v, we can put ALL of N[v] in one clique!
        #
        # But that uses all C-C edges within N[v]. If multiple v's share the same C
        #   vertices, the C-C edges within those shared vertices would be covered
        #   multiple times - which is NOT allowed (each edge in exactly one clique).
        #
        # So the constraint is: for any pair (a,b) in C, the set of v's that have
        #   both a and b in their clique must be at most 1 (since edge (a,b) can only
        #   be in one clique).
        #
        # This is the key structural constraint!
        #
        # Reformulation: For each v in I, we partition N[v] into groups. Each group
        #   S for v creates a clique {v}∪S. The C-C edges within S are "consumed".
        #   No C-C edge can be consumed by two different cliques.
        #
        # Objective: minimize total number of cliques = sum over v of (number of groups
        #   for v) + (number of C-only cliques for residual C-C edges).
        #
        # Key trade-off: For a given v, using 1 group (all of N[v]) uses |N[v]|*(|N[v]|-1)/2
        #   C-C edges. Using more groups uses fewer C-C edges per group but more cliques.
        #
        # Optimal strategy: For each v, we want to use as few groups as possible,
        #   subject to the constraint that no C-C edge is used twice.
        #
        # This is equivalent to: assign each v a partition of N[v] such that for any
        #   pair {a,b} in C, at most one v has a and b in the same group.
        #
        # Simplification: If we give each v exactly ONE group (N[v]), then C-C edge
        #   (a,b) is used by all v that have both a and b in N[v]. This violates
        #   the constraint unless for every pair (a,b), at most one v contains both.
        #
        # So the constraint is: for each pair (a,b) in C, |{v : a,b in N[v]}| <= 1.
        #
        # If this holds, we can give each v one clique {v, N[v]} and the residual
        #   C-C edges are those not in any N[v].
        #
        # If it doesn't hold, we need to split some v's.
        #
        # Let's check: with p=0.7, k=10, n=30, |I|=20. For a pair (a,b), the number
        #   of v with both a,b in N[v] is Binomial(20, 0.49) ~ mean 9.8. So many
        #   violations. We need to split.
        #
        # Better approach: Model as a scheduling/packing problem.
        #
        # For each v, we need to partition N[v] into groups. The "cost" is the number
        #   of groups. The "resource" is C-C edges: each group S consumes all C-C edges
        #   within S.
        #
        # Greedy with lookahead: Process v's in order of |N[v]| descending (largest
        #   first, as they consume the most resources). For each v, find the minimum
        #   number of groups such that no group contains a pair already consumed.
        #
        # This is equivalent to: color the graph on N[v] where two vertices are
        #   adjacent if their C-C edge is already consumed. Minimum coloring = minimum
        #   groups.
        #
        # Let's implement this!

        used_cc = set()  # C-C edges already consumed
        partition = []

        # Process I vertices in order of |N[v]| descending
        I_sorted = sorted(I, key=lambda v: -len(N[v]))

        for v in I_sorted:
            if time.time() > deadline:
                break
            Nv = N[v]
            if not Nv:
                continue

            # Build conflict graph on Nv: edge (a,b) if (a,b) in used_cc
            h_vertices = sorted(Nv)
            h_adj = {u: set() for u in h_vertices}
            for i in range(len(h_vertices)):
                a = h_vertices[i]
                for j in range(i + 1, len(h_vertices)):
                    b = h_vertices[j]
                    e = (min(a, b), max(a, b))
                    if e in used_cc:
                        h_adj[a].add(b)
                        h_adj[b].add(a)

            # Color h_adj: minimum coloring gives minimum groups
            colors = dsatur_coloring(h_vertices, h_adj)

            # Group by color
            color_classes = defaultdict(list)
            for u, c in colors.items():
                color_classes[c].append(u)

            for c in sorted(color_classes.keys()):
                S = sorted(color_classes[c])
                clique = sorted(S + [v])
                partition.append(clique)
                # Mark C-C edges within S as used
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        used_cc.add((min(a, b), max(a, b)))

        # Cover remaining C-C edges with C-only cliques
        remaining_cc = all_c_edges - used_cc

        if remaining_cc:
            # Build graph on C with remaining edges
            rem_adj = {u: set() for u in C}
            for (u, v) in remaining_cc:
                rem_adj[u].add(v)
                rem_adj[v].add(u)

            # Greedy: repeatedly find a maximal clique in remaining graph
            remaining_set = set(remaining_cc)
            while remaining_set:
                if time.time() > deadline:
                    # Fallback: single edges
                    for (u, v) in remaining_set:
                        partition.append([u, v])
                    break

                # Find a large clique greedily
                # Pick vertex with max degree in remaining
                deg = {u: len(rem_adj[u] & set(x for (x, y) in remaining_set for z in [y] if z == u) | set(x for (x, y) in remaining_set for z in [x] if z == u)) for u in C}
                # Simpler: just pick a vertex with high remaining degree
                rem_deg = {u: 0 for u in C}
                for (a, b) in remaining_set:
                    rem_deg[a] += 1
                    rem_deg[b] += 1

                start = max(C, key=lambda u: rem_deg[u])
                # Greedy expand
                clique = {start}
                candidates = set(rem_adj[start])
                while True:
                    added = False
                    for w in sorted(candidates, key=lambda x: -rem_deg[x]):
                        if w in clique:
                            continue
                        # Check w connects to all in clique via remaining edges
                        can_add = all((min(w, z), max(w, z)) in remaining_set for z in clique)
                        if can_add:
                            clique.add(w)
                            added = True
                            break
                    if not added:
                        break

                if len(clique) < 2:
                    # Single edge
                    u, v = next(iter(remaining_set))
                    clique = {u, v}

                partition.append(sorted(clique))

                # Remove all C-C edges within clique
                cl = sorted(clique)
                for i in range(len(cl)):
                    for j in range(i + 1, len(cl)):
                        a, b = cl[i], cl[j]
                        remaining_set.discard((min(a, b), max(a, b)))

        # Convert to list of lists of ints
        final_cliques = []
        for c in partition:
            final_cliques.append([int(x) for x in c])

        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for n, k, p, seed in specs:
        if time.time() > deadline:
            results.append([])
            continue
        result = solve_instance(n, k, p, seed)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
