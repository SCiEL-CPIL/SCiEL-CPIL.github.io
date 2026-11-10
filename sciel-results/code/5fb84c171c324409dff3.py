"""Clique partition of split graphs using structural insight + greedy refinement."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build edge set and adjacency
        edge_set = set()
        for e in edges:
            u, v = e
            if u > v:
                u, v = v, u
            edge_set.add((u, v))

        # Determine which independent vertices are connected to which clique vertices
        # adj[i] for i in [k, n) is a set of clique vertices it's connected to
        indep_adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            if u < k and v >= k:
                indep_adj[v].add(u)
            elif v < k and u >= k:
                indep_adj[u].add(v)

        # Key structural insight for split graphs:
        # Any clique is either:
        # 1. A subset of C (clique part)
        # 2. A subset of C union some vertices from I (independent set) that are connected to all in the C-subset
        #
        # Strategy: 
        # - For each independent vertex v, let N(v) be its neighbors in C
        # - A clique containing v and some clique vertices S must have S subset of N(v)
        # - We want to partition edges using minimum cliques
        #
        # Greedy approach:
        # 1. Handle edges within C: these can all be covered by the full C clique if we include all of C
        #    But we need to be strategic - maybe it's better to split C into pieces
        # 2. For each independent vertex v, edges from v to N(v) can be covered by one clique {v} union N(v)
        #    But if multiple independent vertices share the same N(v), we can combine them? No - they're not connected to each other.
        #
        # Actually, since vertices in I are not connected to each other, any clique can contain at most one vertex from I.
        # So each clique is either:
        # - A subset of C
        # - {v} union S where v in I and S subset of N(v)
        #
        # Edges to cover:
        # - All edges within C
        # - All edges between C and I
        #
        # Since any clique with an I-vertex can only cover edges incident to that I-vertex (and within its C-neighbors),
        # and cliques within C cover C-internal edges, we need to think about this carefully.
        #
        # Observation: If we use the full C as a clique, it covers all C-C edges. Then for each I-vertex v,
        # we need to cover edges from v to N(v). Since v is only connected to N(v), and N(v) is a subset of C,
        # we can cover all edges from v to N(v) with the clique {v} union N(v).
        # But wait - this clique also covers edges within N(v), which are already covered by C.
        # The requirement is that every edge lies in EXACTLY ONE listed clique.
        #
        # So we cannot overlap cliques on edges. We need a partition of edges.
        #
        # Refined strategy:
        # - Don't use the full C clique if it would cause overlap issues
        # - Instead, for each I-vertex v, create a clique {v} union S_v where S_v subset of N(v)
        # - The remaining C-C edges and any C-I edges not covered must be covered by C-only cliques
        #
        # Actually, let's think differently. Each edge must be in exactly one clique.
        # Edges are of two types: C-C edges and C-I edges.
        # A clique of type {v} union S (v in I, S subset C) covers:
        #   - C-I edges between v and each u in S
        #   - C-C edges between pairs in S
        # A clique of type S (S subset C) covers:
        #   - C-C edges between pairs in S
        #
        # To minimize total cliques, we want to cover as many edges per clique as possible.
        # The best cliques are those that include an I-vertex with many C-neighbors, because they cover
        # both C-I and C-C edges in one clique.
        #
        # Greedy algorithm:
        # 1. Track remaining edges
        # 2. At each step, find the clique that covers the most remaining edges
        # 3. For I-vertex v, the best clique is {v} union S where S is chosen to maximize uncovered edges
        #    For C-only, the best clique is a large subset of C with many uncovered internal edges
        #
        # Simplified greedy:
        # For each I-vertex v, compute N(v) = neighbors in C
        # The clique {v} union N(v) covers all edges from v to N(v) plus all edges within N(v)
        # Start with all I-vertices, greedily pick the one whose clique covers the most uncovered edges
        # After processing all I-vertices (or as needed), cover remaining C-C edges with C-cliques

        remaining_edges = set(edge_set)

        # Precompute N(v) for each I-vertex
        N = {}
        for v in range(k, n):
            N[v] = indep_adj[v]  # set of clique vertices connected to v

        # We'll greedily add cliques
        cliques = []

        # First, try to use cliques with I-vertices to cover C-I edges
        # Sort I-vertices by |N(v)| descending
        i_vertices = sorted(range(k, n), key=lambda v: -len(N[v]))

        for v in i_vertices:
            if time.time() > deadline - 10:
                break
            # Find edges from v that are still remaining
            v_edges = [(u, v) if u < v else (v, u) for u in N[v] if (min(u, v), max(u, v)) in remaining_edges]
            if not v_edges:
                continue

            # The clique {v} union N(v) covers all C-I edges from v to N(v) and C-C edges within N(v)
            # But some of those C-C edges might already be covered, and some C-I edges might not be remaining
            # We want to choose S subset of N(v) to maximize uncovered edges covered
            # Simplest: use S = N(v) and see what gets covered
            S = N[v]
            clique = [v] + list(S)
            if len(clique) < 2:
                continue

            # Check how many remaining edges this clique covers
            clique_set = set(clique)
            covered = set()
            for (u, w) in remaining_edges:
                if u in clique_set and w in clique_set:
                    covered.add((u, w))

            if not covered:
                continue

            cliques.append(clique)
            remaining_edges -= covered

        # Now cover remaining edges (should be only C-C edges)
        # For C-C edges, we need to partition them into cliques (subsets of C)
        # Each subset of C is a clique
        # We want to minimize the number of subsets whose pairwise edges cover all remaining C-C edges
        #
        # This is equivalent to edge coloring of a complete-ish graph... but actually it's simpler:
        # Any two C-C edges that share a vertex can be in the same clique if the three endpoints form a clique
        # (which they do since C is a clique). But actually, any subset of C is a clique, so we just need
        # to partition the edges into groups where each group is a set of edges that are all within some subset of C.
        #
        # Actually, the simplest approach: each remaining C-C edge (u, w) can be its own clique [u, w],
        # but we can do better. Any set of edges that form a "clique cover" - but since all C vertices are
        # mutually adjacent, any subset of C is a clique. So we can put multiple edges in one clique as long
        # as all edges have both endpoints in the clique.
        #
        # Greedy: pick a vertex with high degree in remaining C-C edges, extend to a large clique
        if remaining_edges:
            rem_c_edges = set()
            for (u, v) in remaining_edges:
                if u < k and v < k:
                    rem_c_edges.add((min(u, v), max(u, v)))

            if rem_c_edges:
                # Build adjacency for remaining C-C edges
                c_adj = [set() for _ in range(k)]
                for (u, v) in rem_c_edges:
                    c_adj[u].add(v)
                    c_adj[v].add(u)

                remaining_c = set(rem_c_edges)
                while remaining_c:
                    if time.time() > deadline - 5:
                        # Fallback: one edge per clique
                        for (u, v) in remaining_c:
                            cliques.append([u, v])
                        break

                    # Find vertex with max degree in remaining C-C edges
                    c_deg = [0] * k
                    for (u, v) in remaining_c:
                        c_deg[u] += 1
                        c_deg[v] += 1

                    max_v = max(range(k), key=lambda x: c_deg[x])
                    if c_deg[max_v] == 0:
                        # Shouldn't happen
                        break

                    # Greedily build a large clique containing max_v
                    current = [max_v]
                    candidates = list(c_adj[max_v])
                    candidates.sort(key=lambda x: -c_deg[x])

                    for cand in candidates:
                        # Check if cand is connected to all in current
                        can_add = True
                        for vc in current:
                            if cand not in c_adj[vc]:
                                can_add = False
                                break
                        if can_add:
                            current.append(cand)
                            c_adj[cand] = c_adj[cand] & set(current[:-1])

                    if len(current) < 2:
                        # Take any edge
                        u, v = next(iter(remaining_c))
                        current = [u, v]

                    cliques.append(current)

                    # Remove covered edges
                    clique_set = set(current)
                    remaining_c = {(u, v) for (u, v) in remaining_c if not (u in clique_set and v in clique_set)}

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue

        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)

        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])

        results.append(final_cliques)

    return results
# EVOLVE-BLOCK-END
