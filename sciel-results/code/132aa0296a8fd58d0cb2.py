"""Rewritten program using greedy clique partitioning with bitmask adjacency."""
import time
from random import Random


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def partition_cliques(n, edges):
        if not edges:
            return []

        # Build adjacency as sets for fast common-neighbor queries
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Track remaining edges
        edge_set = set(edges)
        remaining = list(edge_set)
        cliques = []

        # Precompute: for each vertex, its neighbors
        # We'll process edges in order of "potential" - edges where both endpoints
        # have high degree in the remaining graph

        while remaining:
            if time.time() > DEADLINE:
                # Emergency: assign remaining edges as individual cliques
                for e in remaining:
                    cliques.append(list(e))
                break

            # Find the best edge to start a clique from
            # Score: number of common neighbors (potential for large clique)
            best_edge = None
            best_score = -1
            best_common = set()

            # Sample edges to find a good one (for speed on large graphs)
            # For small graphs, check all
            sample = remaining if len(remaining) <= 200 else remaining[::max(1, len(remaining)//200)]
            for e in sample:
                u, v = e
                common = adj[u] & adj[v]
                # Filter to only vertices that are still connected via remaining edges
                # Actually, we need common neighbors that form a clique with u,v
                # For now, just count common neighbors as a heuristic
                score = len(common)
                if score > best_score:
                    best_score = score
                    best_edge = e
                    best_common = common

            if best_edge is None:
                best_edge = remaining[0]
                u, v = best_edge
                best_common = adj[u] & adj[v]

            u, v = best_edge
            # Start clique with {u, v}
            clique = [u, v]
            clique_set = {u, v}

            # Try to extend the clique
            # We need to add vertices that are connected to ALL vertices in the clique
            # and that have remaining edges to all clique members
            candidates = set(best_common)
            # Filter: only vertices with remaining edges to u and v
            candidates = {x for x in candidates if (min(u,x), max(u,x)) in edge_set or (min(v,x), max(v,x)) in edge_set}
            # More precisely: x must have remaining edge to u AND to v
            candidates = {x for x in candidates
                          if (min(u,x), max(u,x)) in edge_set
                          and (min(v,x), max(v,x)) in edge_set}

            # Greedily add vertices
            changed = True
            while changed:
                changed = False
                # Find candidate that maximizes connections to existing clique
                best_cand = None
                best_cand_score = -1
                for x in sorted(candidates, key=lambda x: -len(adj[x] & clique_set)):
                    # Check if x is connected to all vertices in clique via remaining edges
                    valid = True
                    for y in clique:
                        a, b = min(x, y), max(x, y)
                        if (a, b) not in edge_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Score: how many other candidates are also connected to x
                    score = len(adj[x] & candidates)
                    if score > best_cand_score:
                        best_cand_score = score
                        best_cand = x
                if best_cand is not None:
                    clique.append(best_cand)
                    clique_set.add(best_cand)
                    # Remove best_cand from candidates
                    candidates.discard(best_cand)
                    # Update candidates: must be connected to best_cand via remaining edge
                    candidates = {x for x in candidates
                                  if (min(x, best_cand), max(x, best_cand)) in edge_set}
                    changed = True

            # Remove all edges within the clique from remaining
            clique_edges = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = min(clique[i], clique[j]), max(clique[i], clique[j])
                    if (a, b) in edge_set:
                        clique_edges.add((a, b))

            # Only add clique if it has at least 2 vertices and covers at least one edge
            if len(clique) >= 2 and clique_edges:
                cliques.append(clique)
                edge_set -= clique_edges
                remaining = [e for e in remaining if e not in clique_edges]
            else:
                # Fallback: just use the edge itself
                if best_edge in edge_set:
                    cliques.append(list(best_edge))
                    edge_set.discard(best_edge)
                    remaining = [e for e in remaining if e != best_edge]

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = partition_cliques(n, edges)
        results.append(cliques)

    return results
