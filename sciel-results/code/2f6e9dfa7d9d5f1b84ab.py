"""Program for clique partition of two dense random graphs.

Strategy: greedy maximum-clique extraction with smart seed selection.
Every edge is placed in exactly one clique; each clique has >= 2 vertices.
Minimizes total number of cliques by maximizing edges per clique.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def _random_graph(n, p, seed):
    """Build the edge list exactly as the verifier does."""
    from random import Random
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return edges


def _greedy_clique_partition(adj, n):
    """
    Greedily partition edges into cliques.
    At each step, pick the seed vertex that can form the largest clique,
    then grow the clique greedily by adding the vertex that covers the
    most remaining edges.
    """
    # Adjacency sets for O(1) lookup
    adj_sets = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                adj_sets[u].add(v)
                adj_sets[v].add(u)

    # Remaining edges
    edge_set = set()
    for u in range(n):
        for v in adj_sets[u]:
            if u < v:
                edge_set.add((u, v))

    # Remaining degree
    rem_deg = [len(adj_sets[u]) for u in range(n)]

    cliques = []

    while edge_set:
        # Find the best seed: vertex with remaining edges that can form
        # the largest possible clique
        best_seed = None
        best_clique_size = 0
        best_potential = 0

        # Try each vertex with remaining edges
        for seed in range(n):
            if rem_deg[seed] == 0:
                continue

            # Greedily grow a clique from seed to estimate its size
            clique = {seed}
            # Candidates: neighbors of all clique members with remaining edges
            candidates = set(adj_sets[seed])
            candidates = {v for v in candidates if rem_deg[v] > 0}

            while candidates:
                # Pick candidate that maximizes edges to current clique
                best_cand = None
                best_score = -1
                for v in candidates:
                    # Count remaining edges from v to clique members
                    score = 0
                    for c in clique:
                        if (min(c, v), max(c, v)) in edge_set:
                            score += 1
                    # Also consider edges from v to other candidates (potential)
                    if score > best_score:
                        best_score = score
                        best_cand = v

                if best_cand is None:
                    break

                clique.add(best_cand)
                # Update candidates: must be adjacent to all clique members
                new_candidates = set()
                for v in candidates:
                    if v == best_cand:
                        continue
                    ok = True
                    for c in clique:
                        if c not in adj_sets[v]:
                            ok = False
                            break
                    if ok and rem_deg[v] > 0:
                        new_candidates.add(v)
                candidates = new_candidates

            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_seed = seed
                best_potential = best_clique_size

        if best_seed is None:
            break

        # Now actually extract the clique from best_seed
        seed = best_seed
        clique = [seed]
        candidates = set(adj_sets[seed])
        candidates = {v for v in candidates if rem_deg[v] > 0}

        while candidates:
            # Pick candidate that maximizes remaining edges to clique
            best_cand = None
            best_score = -1
            for v in candidates:
                score = 0
                for c in clique:
                    if (min(c, v), max(c, v)) in edge_set:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = v

            if best_cand is None:
                break

            clique.append(best_cand)
            # Update candidates
            new_candidates = set()
            for v in candidates:
                if v == best_cand:
                    continue
                ok = True
                for c in clique:
                    if c not in adj_sets[v]:
                        ok = False
                        break
                if ok and rem_deg[v] > 0:
                    new_candidates.add(v)
            candidates = new_candidates

        # Remove all edges among clique members
        added = False
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    edge_set.remove(e)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                    added = True

        if len(clique) >= 2 and added:
            cliques.append(clique)

    return cliques


def _partition_one(n, p, seed):
    edges = _random_graph(n, p, seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for (u, v) in edges:
        adj[u, v] = 1
        adj[v, u] = 1

    cliques = _greedy_clique_partition(adj, n)

    return [[int(x) for x in cl] for cl in cliques]


def construct():
    res1 = _partition_one(32, 0.6, 601)
    res2 = _partition_one(64, 0.55, 602)
    return [res1, res2]
# EVOLVE-BLOCK-END
