"""Clique partition of split graphs exploiting structural properties."""
import time
import random
from collections import defaultdict


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # Build cross-neighborhoods: for each i in I, which C vertices are adjacent
        C = list(range(k))
        I = list(range(k, n))

        # For each i in I, get its neighbors in C
        cross_neighbors = {}
        for i in I:
            cross_neighbors[i] = set()
        for u, v in edges:
            if u < k and v >= k:
                cross_neighbors[v].add(u)
            elif v < k and u >= k:
                cross_neighbors[u].add(v)

        # Edges within C
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        # Strategy:
        # 1. For each i in I, partition N(i) into cliques (subsets of C).
        #    Each subset S gives clique {i} ∪ S.
        #    These cliques cover edges (i, s) for s in S AND edges (s1, s2) for s1,s2 in S.
        # 2. Track which C-edges are covered by these cross-cliques.
        # 3. Partition remaining C-edges into C-only cliques.
        #
        # To minimize total cliques, we want large subsets for each i.
        # Since C is a clique, any subset of C is a clique.
        # So for each i, N(i) can be one clique: {i} ∪ N(i).
        # But that uses 1 clique per i and covers all cross edges for i.
        # However, it also covers ALL edges within N(i) in C.
        #
        # Alternative: split N(i) into smaller pieces to reduce overlap of C-edge coverage?
        # No - we want to COVER C-edges, not avoid them. More C-edges covered by cross-cliques
        # means fewer C-only cliques needed.
        #
        # So optimal: for each i, use ONE clique {i} ∪ N(i) (if N(i) non-empty).
        # This covers all cross edges for i and all C-edges within N(i).
        #
        # Then remaining C-edges = C_edges - union of all C-edges within N(i) for i in I.
        # Partition remaining C-edges into cliques.
        #
        # Wait - but {i} ∪ N(i) might be very large. That's fine, it's still one clique.
        # Total cliques = |{i in I : N(i) != empty}| + (cliques to cover remaining C-edges)
        #
        # Can we do better? What if we DON'T use a single clique for i, but split N(i)?
        # Splitting N(i) into S1, S2 gives 2 cliques instead of 1, but covers the same C-edges
        # (edges within S1 and within S2, but NOT edges between S1 and S2).
        # So splitting REDUCES C-edge coverage, which is bad.
        # Therefore: use one clique per i with non-empty N(i).
        #
        # Actually wait - we need each clique to have >= 2 vertices.
        # If N(i) = {u} (single vertex), then {i, u} is a valid clique (2 vertices).
        # If N(i) is empty, no clique needed for i.
        #
        # So: cliques = [{i} ∪ N(i) for i in I if N(i)] + partition of remaining C-edges
        #
        # For partitioning remaining C-edges: C is a k-clique, so any subset is a clique.
        # We need to partition the remaining edge set into cliques.
        # This is the problem of partitioning a graph (which is a subgraph of K_k) into cliques.
        # Since the remaining graph is a subgraph of K_k, we can use greedy:
        # repeatedly take a maximal clique in the remaining graph.

        cliques = []

        # Step 1: Cross-cliques for each i in I with non-empty N(i)
        covered_C_edges = set()
        for i in I:
            Ni = cross_neighbors[i]
            if not Ni:
                continue
            clique = sorted([i] + list(Ni))
            cliques.append(clique)
            # Mark all C-edges within Ni as covered
            Ni_list = sorted(Ni)
            for a_idx in range(len(Ni_list)):
                for b_idx in range(a_idx + 1, len(Ni_list)):
                    covered_C_edges.add((Ni_list[a_idx], Ni_list[b_idx]))

        # Step 2: Remaining C-edges
        remaining_C_edges = C_edges - covered_C_edges

        # Partition remaining C-edges into cliques
        # The remaining graph is on vertices C, with edges = remaining_C_edges
        # Use greedy: find maximal cliques
        remaining = set(remaining_C_edges)
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        while remaining:
            if time.time() > deadline:
                # Fallback: each edge as its own clique
                for u, v in remaining:
                    cliques.append(sorted([u, v]))
                break

            # Find a maximal clique in the remaining graph
            # Greedy: pick vertex with highest degree, expand
            if not rem_adj:
                break

            # Pick vertex with max degree
            max_deg = 0
            start_v = None
            for v in rem_adj:
                if len(rem_adj[v]) > max_deg:
                    max_deg = len(rem_adj[v])
                    start_v = v

            if start_v is None:
                break

            # Greedy clique expansion
            clique = {start_v}
            candidates = set(rem_adj[start_v])
            while candidates:
                # Pick candidate with max connections to current clique
                best_c = None
                best_score = -1
                for c in candidates:
                    score = len(clique & rem_adj[c])
                    if score > best_score:
                        best_score = score
                        best_c = c
                if best_c is None:
                    break
                clique.add(best_c)
                # Update candidates: must be adjacent to all in clique
                candidates = set()
                for c in candidates:
                    if all((min(c, q), max(c, q)) in remaining for q in clique):
                        candidates.add(c)

            if len(clique) < 2:
                # Just take one edge
                e = next(iter(remaining))
                clique = set(e)

            cliques.append(sorted(clique))

            # Remove all edges in this clique from remaining
            clique_list = sorted(clique)
            for a_idx in range(len(clique_list)):
                for b_idx in range(a_idx + 1, len(clique_list)):
                    remaining.discard((clique_list[a_idx], clique_list[b_idx]))

            # Update rem_adj
            rem_adj.clear()
            for u, v in remaining:
                rem_adj[u].add(v)
                rem_adj[v].add(u)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
