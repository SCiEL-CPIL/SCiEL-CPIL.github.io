"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        deadline = time.time() + 140.0
        rng = Random(seed)

        # Generate cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))

        # Build adjacency for independent vertices
        indep_neighbors = {v: set() for v in range(k, n)}
        for (u, v) in cross_edges:
            indep_neighbors[v].add(u)

        # All edges in the graph
        all_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_edges.add((u, v))
        for (u, v) in cross_edges:
            all_edges.add((min(u, v), max(u, v)))

        # Strategy: Greedy set cover on edges.
        # A clique can be:
        #   Type 1: subset of [0, k) with size >= 2 (covers internal edges)
        #   Type 2: {v} ∪ S where v in [k, n), S subset of [0, k), S nonempty
        #           (covers cross edges (v, u) for u in S, and internal edges within S)
        #   Type 3: {v, u} where v in [k, n), u in [0, k) (special case of Type 2 with |S|=1)
        #
        # Key constraint: each edge in exactly one clique.
        #
        # Greedy approach:
        # 1. Start with all edges uncovered.
        # 2. Repeatedly pick the clique that covers the most uncovered edges.
        # 3. Add it, mark those edges as covered.
        # 4. Stop when all edges are covered.
        #
        # To make this efficient, we consider:
        # - For each independent vertex v, the best S ⊆ neighbors(v) to use.
        #   The clique {v} ∪ S covers: cross edges (v,u) for u in S, plus internal edges within S.
        #   But internal edges within S might already be covered or might conflict.
        # - Pure internal cliques: subsets of [0,k).
        #
        # Simplified greedy:
        # Phase 1: For each independent vertex v, assign it a clique {v} ∪ S_v where
        #          S_v is chosen to maximize newly covered edges.
        #          Process vertices in order of decreasing degree.
        # Phase 2: Cover remaining internal edges with minimal cliques.
        #
        # Since internal edges form a complete graph on [0,k), and some are already
        # covered by Phase 1 cliques, we need to cover the rest.
        # The remaining internal edges form a graph on [0,k). We can cover them with
        # one clique per connected component... no, that's not right.
        # Actually, if we have uncovered internal edges, we can just add the set of
        # vertices incident to uncovered internal edges as one clique (if size >= 2).
        # But that might re-cover some edges. So we need to be careful.
        #
        # Better approach: Track which internal edges are covered.
        # After Phase 1, the uncovered internal edges form a subgraph. We need to
        # cover it with cliques that don't include any independent vertices (to avoid
        # re-covering cross edges) and don't re-cover already-covered internal edges.
        #
        # Actually, the cleanest approach:
        # - Use a greedy algorithm that at each step evaluates candidate cliques.
        # - Candidate cliques: for each independent vertex v, {v} ∪ N(v) (full neighborhood),
        #   and also smaller subsets. Plus internal cliques.
        # - This is expensive, so let's use a heuristic.

        # Simpler structural approach:
        # 1. For each independent vertex v, create clique {v} ∪ N(v).
        #    This covers all cross edges incident to v and all internal edges within N(v).
        # 2. Track which internal edges are now covered.
        # 3. For remaining uncovered internal edges, we need to cover them.
        #    The remaining internal edges are those not within any N(v).
        #    Let U = set of vertices u in [0,k) such that (u, w) is an uncovered internal edge for some w.
        #    Add U as a clique if |U| >= 2. But this might re-cover some edges!
        #
        # The problem with step 3 is that adding U as a clique covers ALL edges within U,
        # including some that were already covered by step 1 cliques. This violates the
        # "exactly one" constraint.
        #
        # So we need a different approach for step 3.
        #
        # Alternative: After step 1, the uncovered internal edges form a graph G_rem on [0,k).
        # We need to partition the edges of G_rem into cliques. Since G_rem is a subgraph of
        # a complete graph, we can cover it by: for each connected component, if it's a clique,
        # use one clique; otherwise, we need more.
        #
        # Actually, the simplest valid approach that allows packing:
        # 1. Don't use a big internal clique.
        # 2. For each independent vertex v, use clique {v} ∪ N(v).
        # 3. For internal edges not covered by any {v} ∪ N(v), cover them separately.
        #    An internal edge (u, w) is covered by {v} ∪ N(v) iff u, w ∈ N(v).
        #    So (u, w) is uncovered iff there is no v with u, w ∈ N(v).
        #    Let F = {(u, w) : u < w, both in [0,k), and no v has u,w ∈ N(v)}.
        #    We need to cover F with cliques. Since F is a subset of the complete graph on [0,k),
        #    we can cover F by: for each maximal clique in F, add it. But F might not be a union
        #    of cliques in a simple way.
        #
        # Even simpler: cover each edge in F as its own clique {u, w}.
        # Total cliques: (number of independent vertices with nonempty N(v)) + |F|.
        #
        # This is valid because:
        # - Each {v} ∪ N(v) covers cross edges (v, u) for u ∈ N(v) and internal edges within N(v).
        # - Each {u, w} for (u,w) ∈ F covers just that one internal edge.
        # - No edge is in two cliques: cross edges are only in their {v} ∪ N(v) clique.
        #   Internal edges are either in exactly one {v} ∪ N(v) (if covered in step 2) or
        #   in exactly one {u, w} clique (if in F).
        #
        # Let's verify: Can an internal edge (u, w) be in two different {v1} ∪ N(v1) and {v2} ∪ N(v2)?
        # Yes! If u, w ∈ N(v1) AND u, w ∈ N(v2), then (u,w) is in both cliques. INVALID.
        #
        # So we need to ensure that each internal edge is covered by at most one {v} ∪ N(v) clique.
        # This means: for any pair (u, w) in [0,k), at most one v can have both u and w in N(v).
        #
        # This is a strong constraint. With p=0.7 and k=10,30,60, many pairs will be in multiple N(v).
        #
        # So the "use {v} ∪ N(v) for all v" approach doesn't work directly.
        #
        # Revised approach:
        # 1. For each independent vertex v, we assign a subset S_v ⊆ N(v).
        # 2. The cliques are {v} ∪ S_v for each v with S_v nonempty.
        # 3. Constraint: for any pair (u, w) in [0,k), at most one v has u, w ∈ S_v.
        #    (This ensures internal edges are covered at most once by these cliques.)
        # 4. Remaining internal edges (not covered by any {v} ∪ S_v) are covered by separate {u, w} cliques.
        #
        # To maximize packing (minimize total cliques), we want S_v to be as large as possible
        # while maintaining the constraint.
        #
        # Greedy: Process independent vertices in some order. For each v, choose S_v ⊆ N(v)
        # to maximize |S_v| subject to: for all pairs (u,w) in S_v, no previously assigned
        # S_{v'} contains both u and w.
        #
        # This is still complex. Let me try an even simpler approach:
        #
        # Approach: For each independent vertex v, S_v = N(v) ∩ R_v where R_v is the set of
        # clique-part vertices not yet "claimed" by previous vertices in a pairwise sense.
        #
        # Actually, let me just go with the safe approach that's known to work:
        # 1. Big internal clique [0, k) if k >= 2.
        # 2. For each cross edge (u, v), clique {u, v}.
        # Total: 1 + |cross_edges| (if k >= 2) or |cross_edges| (if k < 2).
        #
        # This is the current approach. To improve, I need to pack cross edges.
        #
        # The ONLY way to pack cross edges without violating the constraint is to NOT use
        # the big internal clique, and instead cover internal edges via the cross-cliques
        # and additional small cliques.
        #
        # Let me implement a greedy approach:
        # - Maintain set of uncovered edges.
        # - Candidate cliques: for each independent vertex v, and for each subset S ⊆ N(v),
        #   the clique {v} ∪ S. Also, for each pair (u,w) in [0,k), the clique {u,w}.
        # - Greedily pick the candidate that covers the most uncovered edges.
        #
        # This is O(n * 2^k * E) which is too slow for k=30.
        #
        # Heuristic: For each independent vertex v, only consider S = N(v) (full neighborhood)
        # and S = {u} for each u in N(v) (single element). Plus internal pairs.
        #
        # Let me implement this with a deadline guard.

        # Build edge list with indices for tracking
        # Use a set of uncovered edges
        uncovered = set(all_edges)
        cliques = []

        # Precompute for each independent vertex its neighbors
        # We'll greedily assign cliques

        # Strategy:
        # 1. For each independent vertex v (in order of decreasing |N(v)|),
        #    try to form clique {v} ∪ S where S ⊆ N(v) maximizing newly covered edges.
        #    S is chosen as the largest subset of N(v) such that no pair in S has its
        #    internal edge already covered by a previously chosen clique.
        #    If S is nonempty, add {v} ∪ S and remove covered edges.
        # 2. Cover remaining edges with pairs.

        # Track which internal edges are covered
        internal_covered = set()

        # Sort independent vertices by degree (descending)
        indep_vertices = [v for v in range(k, n) if indep_neighbors[v]]
        indep_vertices.sort(key=lambda v: len(indep_neighbors[v]), reverse=True)

        for v in indep_vertices:
            if time.time() > deadline:
                break
            Nv = indep_neighbors[v]
            if not Nv:
                continue

            # Find a maximal subset S ⊆ Nv such that for all pairs (u,w) in S,
            # the internal edge (min(u,w), max(u,w)) is not in internal_covered.
            # Greedy: iteratively remove vertices that conflict.
            S = set(Nv)
            changed = True
            while changed:
                changed = False
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        e = (s_list[i], s_list[j])
                        if e in internal_covered:
                            # Conflict: remove s_list[j] (the one with higher index for determinism)
                            S.discard(s_list[j])
                            changed = True
                            break
                    if changed:
                        break

            if len(S) >= 1:
                # Add clique {v} ∪ S
                clique = sorted([v] + list(S))
                cliques.append(clique)

                # Mark cross edges (v, u) for u in S as covered
                for u in S:
                    e = (min(v, u), max(v, u))
                    uncovered.discard(e)

                # Mark internal edges within S as covered
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        e = (s_list[i], s_list[j])
                        internal_covered.add(e)
                        uncovered.discard(e)
            # If S is empty, no clique formed for this v; its cross edges remain uncovered
            # and will be handled by the fallback pair coverage below.

        # Cover remaining uncovered edges
        # These should all be internal edges (cross edges should be covered)
        # For each remaining uncovered edge, add it as a pair clique
        for e in list(uncovered):
            if time.time() > deadline:
                break
            u, w = e
            cliques.append([u, w])
            uncovered.discard(e)

        # Validate: ensure no clique has size < 2
        cliques = [c for c in cliques if len(c) >= 2]

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(build_partition(*spec))

    return results
# EVOLVE-BLOCK-END
