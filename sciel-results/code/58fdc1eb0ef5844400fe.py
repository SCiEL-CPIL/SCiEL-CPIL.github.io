import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def maximal_clique_greedy(edge_mask, candidates_mask, k):
        """Find a maximal clique greedily in the subgraph induced by candidates_mask."""
        if candidates_mask == 0:
            return set()
        # Pick vertex with max degree in candidates
        best_v = None
        best_deg = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = bin(edge_mask[v] & candidates_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
        clique = {best_v}
        common = edge_mask[best_v] & candidates_mask
        while True:
            best_cand = None
            best_score = -1
            temp = common
            while temp:
                lsb = temp & (-temp)
                w = lsb.bit_length() - 1
                temp ^= lsb
                score = bin(edge_mask[w] & common).count('1')
                if score > best_score:
                    best_score = score
                    best_cand = w
            if best_cand is None:
                break
            clique.add(best_cand)
            common &= edge_mask[best_cand]
        return clique

    def remove_clique_edges(edge_mask, clique):
        """Remove all edges within clique from edge_mask."""
        cl = sorted(clique)
        for i in range(len(cl)):
            u = cl[i]
            for j in range(i + 1, len(cl)):
                w = cl[j]
                edge_mask[u] &= ~(1 << w)
                edge_mask[w] &= ~(1 << u)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency for C vertices
        adj_C = [set() for _ in range(k)]
        for u, v in edges:
            if u < k and v < k:
                adj_C[u].add(v)
                adj_C[v].add(u)

        # Build cross adjacency: for each I vertex, its neighbors in C
        cross_adj = [set() for _ in range(n)]
        for u, v in edges:
            if u < k <= v:
                cross_adj[v].add(u)
                cross_adj[u].add(v)
            elif v < k <= u:
                cross_adj[u].add(v)
                cross_adj[v].add(u)

        # Bitmask for C edges
        edge_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj_C[u]:
                    edge_mask[u] |= (1 << v)
                    edge_mask[v] |= (1 << u)

        partition = []

        # Group I vertices by their neighborhood in C
        # Key: frozenset of neighbors in C
        groups = {}
        for v in range(k, n):
            nbrs = frozenset(cross_adj[v])
            if not nbrs:
                continue
            if nbrs not in groups:
                groups[nbrs] = []
            groups[nbrs].append(v)

        # Sort groups by size descending (process larger groups first)
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

        for nbrs_frozen, verts in sorted_groups:
            nbrs = set(nbrs_frozen)
            nbrs_mask = 0
            for u in nbrs:
                nbrs_mask |= (1 << u)

            # Find cliques in the induced subgraph on nbrs
            # Each clique S in nbrs gives us |verts| cliques: {v} ∪ S for each v in verts
            # But we can do better: if we find a clique cover of the edges within nbrs,
            # each clique S in the cover generates |verts| cliques.
            # The total cross cliques = sum over cover cliques S of |verts|
            # But we also remove edges from C that are within each S.

            # Strategy: find a clique cover of the subgraph induced by nbrs.
            # For each clique S in the cover, for each v in verts, add {v} ∪ S to partition.
            # This covers all cross edges from verts to nbrs (since nbrs is the full neighborhood).
            # And it removes the edges within each S from C.

            # Find clique cover of subgraph on nbrs
            remaining_mask = nbrs_mask
            while remaining_mask:
                clique = maximal_clique_greedy(edge_mask, remaining_mask, k)
                if not clique:
                    break
                # For each v in verts, create clique {v} ∪ clique
                for v in verts:
                    clique_list = sorted(list(clique | {v}))
                    partition.append(clique_list)
                # Remove edges within clique from edge_mask
                remove_clique_edges(edge_mask, clique)
                # Remove clique from remaining
                clique_mask = 0
                for u in clique:
                    clique_mask |= (1 << u)
                remaining_mask &= ~clique_mask

        # Remaining edges in C: partition into cliques
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if edge_mask[u] & (1 << v):
                    remaining_edges.append((u, v))

        if not remaining_edges:
            return partition

        # Greedy: repeatedly pick edge with max common neighborhood, extend to maximal clique
        while remaining_edges:
            best_edge = None
            best_common = -1
            for e in remaining_edges:
                u, v = e
                common_mask = edge_mask[u] & edge_mask[v]
                common_count = bin(common_mask).count('1')
                if common_count > best_common:
                    best_common = common_count
                    best_edge = e

            if best_edge is None:
                best_edge = remaining_edges[0]

            u, v = best_edge
            # Build maximal clique containing this edge
            candidates = edge_mask[u] & edge_mask[v] | (1 << u) | (1 << v)
            clique = maximal_clique_greedy(edge_mask, candidates, k)
            # Ensure u and v are in clique
            clique.add(u)
            clique.add(v)

            clique_list = sorted(list(clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)

            remove_clique_edges(edge_mask, clique)

            clique_set = set(clique)
            new_remaining = []
            for e in remaining_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_remaining.append(e)
            remaining_edges = new_remaining

        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results
