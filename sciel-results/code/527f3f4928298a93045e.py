import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        I_set = set(I)

        # Precompute: for each I vertex, its neighbors in C
        I_neigh = {}
        for v in I:
            I_neigh[v] = adj[v] & set(C)

        # We need to partition all edges into cliques.
        # Strategy:
        # 1. For each I vertex v, its edges go to I_neigh[v] (subset of C).
        #    These edges must be covered by cliques that contain v and a clique-subset of I_neigh[v].
        # 2. Edges within C must also be covered.
        #
        # Key insight: A clique in a split graph can contain at most one I vertex.
        # So cliques are either:
        #   - A clique entirely within C
        #   - {v} ∪ S where v ∈ I and S is a clique in C (and S ⊆ I_neigh[v])
        #
        # We want to minimize total number of cliques.
        #
        # Approach: Use a greedy set-cover-like approach.
        # For each I vertex v, we need to cover the edges from v to I_neigh[v].
        # Each clique {v} ∪ S covers all edges from v to vertices in S (that exist).
        # Since S must be a clique in C, and all edges from v to S must exist (S ⊆ I_neigh[v]),
        # we need S to be a clique in the induced subgraph of C on I_neigh[v].
        #
        # For C edges: after assigning I-vertex cliques, remaining C-edges need C-cliques.
        #
        # Better approach: Think of it as covering all edges.
        # Each clique covers a set of edges. We want minimum cliques.
        #
        # Greedy: Repeatedly pick the clique that covers the most uncovered edges.
        # But that's expensive. Let's use a smarter heuristic.
        #
        # For I vertices: group them. If multiple I vertices have the same neighborhood in C,
        # they can share the same C-clique structure. But each I vertex needs its own clique
        # (since two I vertices are not adjacent, they can't be in the same clique).
        #
        # So each I vertex v needs at least 1 clique (if it has any edges).
        # The question is how many C-cliques we need for the C-edges, and how to
        # structure the I-vertex cliques to also help cover C-edges.
        #
        # Wait - a clique {v} ∪ S covers edges: (v, s) for s in S, AND (s1, s2) for s1,s2 in S.
        # So the C-edges within S are also covered!
        #
        # This means we can use I-vertex cliques to help cover C-edges.
        #
        # Strategy:
        # 1. For each I vertex v, find a maximal clique in C induced by I_neigh[v].
        #    Use this as the clique {v} ∪ S_v.
        # 2. After all I-vertex cliques are assigned, remaining C-edges need additional C-cliques.
        # 3. To minimize total, we want the S_v sets to collectively cover as many C-edges as possible.
        #
        # Refined strategy:
        # - Process I vertices. For each, find a large clique in its C-neighborhood.
        # - Track which C-edges are covered.
        # - At the end, cover remaining C-edges with greedy C-cliques.
        #
        # To maximize coverage, process I vertices with larger neighborhoods first,
        # and for each, find a maximal clique that covers many currently-uncovered C-edges.

        # Track C-edges coverage using a set of frozensets
        covered_C_edges = set()

        # For efficiency, track uncovered C-edges
        all_C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_C_edges.add((u, v))

        uncovered_C_edges = set(all_C_edges)

        # Precompute adjacency within C using bitmasks for speed
        C_adj_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in all_C_edges:
                    C_adj_mask[u] |= (1 << v)
                    C_adj_mask[v] |= (1 << u)

        partition = []

        # Sort I vertices by neighborhood size descending (process larger ones first)
        I_sorted = sorted(I, key=lambda v: len(I_neigh[v]), reverse=True)

        for v in I_sorted:
            N_v = I_neigh[v]
            if not N_v:
                continue

            # Find a maximal clique in C induced by N_v
            # Use greedy: start with vertex of max degree in N_v, then greedily add
            nv_list = sorted(N_v)
            if not nv_list:
                continue

            # Build mask for N_v
            nv_mask = 0
            for u in nv_list:
                nv_mask |= (1 << u)

            # Find vertex with max degree within N_v
            best_start = None
            best_deg = -1
            for u in nv_list:
                deg = bin(C_adj_mask[u] & nv_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_start = u

            clique = {best_start}
            common = C_adj_mask[best_start] & nv_mask

            while True:
                best_candidate = None
                best_score = -1
                temp = common
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    score = bin(C_adj_mask[w] & common).count('1')
                    # Also score by how many uncovered C-edges it helps cover
                    # For now, just use degree
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                if best_candidate is None:
                    break
                clique.add(best_candidate)
                common &= C_adj_mask[best_candidate]

            # Add this clique
            clique_list = sorted([v] + list(clique))
            partition.append(clique_list)

            # Mark C-edges within clique as covered
            clique_C = sorted(clique)
            for i in range(len(clique_C)):
                for j in range(i + 1, len(clique_C)):
                    edge = (clique_C[i], clique_C[j])
                    covered_C_edges.add(edge)
                    uncovered_C_edges.discard(edge)

        # Now cover remaining C-edges
        if uncovered_C_edges:
            # Greedy: repeatedly find a maximal clique in the remaining C-edge graph
            remaining = set(uncovered_C_edges)

            while remaining:
                # Find a maximal clique in the graph defined by remaining edges
                # Use edge with max common neighborhood as seed
                best_edge = None
                best_common_count = -1

                for e in remaining:
                    u, v = e
                    common = C_adj_mask[u] & C_adj_mask[v]
                    # Only count vertices that are endpoints of remaining edges
                    # Simplified: just count common neighbors
                    count = bin(common).count('1')
                    if count > best_common_count:
                        best_common_count = count
                        best_edge = e

                if best_edge is None:
                    best_edge = next(iter(remaining))

                u, v = best_edge
                clique = {u, v}
                common = C_adj_mask[u] & C_adj_mask[v]

                # Greedily extend
                while True:
                    best_candidate = None
                    best_score = -1
                    temp = common
                    while temp:
                        lsb = temp & (-temp)
                        w = lsb.bit_length() - 1
                        temp ^= lsb
                        score = bin(C_adj_mask[w] & common).count('1')
                        if score > best_score:
                            best_score = score
                            best_candidate = w
                    if best_candidate is None:
                        break
                    clique.add(best_candidate)
                    common &= C_adj_mask[best_candidate]

                clique_list = sorted(clique)
                if len(clique_list) >= 2:
                    partition.append(clique_list)

                # Remove covered edges from remaining
                clique_set = set(clique)
                new_remaining = set()
                for e in remaining:
                    if e[0] in clique_set and e[1] in clique_set:
                        continue
                    new_remaining.add(e)
                remaining = new_remaining

        # Verify and clean up
        result = []
        for clique in partition:
            if len(clique) >= 2:
                result.append([int(x) for x in sorted(clique)])

        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
