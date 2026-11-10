"""Clique partition of two fixed dense random graphs via optimized bitmask greedy with restarts."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 115.0  # Wall-clock guard


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _encode_edge(u, v):
    if u > v:
        u, v = v, u
    return u * 100 + v


def _decode_edge(e):
    u = e // 100
    v = e % 100
    return u, v


def _greedy_partition_fast(edges, n, rng):
    """
    Fast greedy clique partition using bitmasks.
    Strategy:
    1. Pick an edge with fewest common neighbors (hardest to cover) from a sample.
    2. Extend to a maximal clique using a max-degree heuristic within common neighbors.
    3. Remove all edges in the clique.
    """
    if not edges:
        return []

    # Build adjacency as bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add(_encode_edge(u, v))

    remaining = set(edge_set)
    cliques = []

    # Track degrees in the remaining graph
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    while remaining:
        if time.time() > _DEADLINE - 2:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                u, v = _decode_edge(e)
                cliques.append([u, v])
            break

        # Find candidate edge: pick one with fewest common neighbors (hardest to cover)
        # Sample a few edges to find a good candidate for efficiency
        if len(remaining) > 50:
            sample = rng.sample(list(remaining), 50)
        else:
            sample = list(remaining)

        best_e = None
        best_common_count = float('inf')
        for e in sample:
            u, v = _decode_edge(e)
            common = adj[u] & adj[v]
            count = bin(common).count('1')
            if count < best_common_count:
                best_common_count = count
                best_e = e

        if best_e is None:
            best_e = next(iter(remaining))

        u, v = _decode_edge(best_e)
        C = [u, v]
        cand_mask = adj[u] & adj[v]

        # Extend clique greedily: pick vertex with max degree within candidates
        while cand_mask:
            best_u = -1
            max_deg_u = -1
            temp_mask = cand_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                cand = lsb.bit_length() - 1
                d = bin(adj[cand] & cand_mask).count('1')
                if d > max_deg_u:
                    max_deg_u = d
                    best_u = cand
                temp_mask ^= lsb

            if best_u == -1:
                break

            C.append(best_u)
            cand_mask &= adj[best_u]
            # Remove vertices already in C
            for w in C:
                cand_mask &= ~(1 << w)

        if len(C) >= 2:
            cliques.append(C)
            # Remove all edges within C from remaining and update adj/deg
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u2, w = C[i], C[j]
                    e = _encode_edge(u2, w)
                    if e in remaining:
                        remaining.discard(e)
                        adj[u2] &= ~(1 << w)
                        adj[w] &= ~(1 << u2)
                        deg[u2] -= 1
                        deg[w] -= 1
        else:
            # Fallback: take one edge
            e = best_e
            u2, w = _decode_edge(e)
            cliques.append([u2, w])
            remaining.discard(e)
            adj[u2] &= ~(1 << w)
            adj[w] &= ~(1 << u2)
            deg[u2] -= 1
            deg[w] -= 1

    return cliques


def _local_improve(cliques, edges, n, rng):
    """
    Try to merge pairs of cliques if their union is also a clique.
    Repeat until no more merges are possible or time is up.
    """
    if time.time() > _DEADLINE - 5:
        return cliques

    # Build adjacency for the full graph (for clique checking)
    adj_full = [0] * n
    for u, v in edges:
        adj_full[u] |= (1 << v)
        adj_full[v] |= (1 << u)

    improved = True
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        # Try all pairs
        for i in range(len(cliques)):
            if time.time() > _DEADLINE - 5:
                return cliques
            for j in range(i + 1, len(cliques)):
                if time.time() > _DEADLINE - 5:
                    return cliques
                merged = list(cliques[i]) + list(cliques[j])
                # Check if merged is a clique
                is_clique = True
                for k in range(len(merged)):
                    for l in range(k + 1, len(merged)):
                        u, v = merged[k], merged[l]
                        if not (adj_full[u] >> v) & 1:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    # Merge them
                    new_cliques = []
                    for idx in range(len(cliques)):
                        if idx != i and idx != j:
                            new_cliques.append(cliques[idx])
                    new_cliques.append(merged)
                    cliques = new_cliques
                    improved = True
                    break
            if improved:
                break

    return cliques


def _validate_partition(cliques, edges):
    """Validate that cliques form a valid partition of edges."""
    edge_set = set(_encode_edge(u, v) for u, v in edges)
    seen_edges = set()
    for c in cliques:
        k = len(c)
        if k < 2:
            return False
        for i in range(k):
            for j in range(i + 1, k):
                e = _encode_edge(c[i], c[j])
                if e not in edge_set or e in seen_edges:
                    return False
                seen_edges.add(e)
    return len(seen_edges) == len(edges)


def _solve_instance(edges, n):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    best_cliques = None
    best_count = float('inf')

    # Run restarts until deadline
    restart_count = 0
    while time.time() < _DEADLINE - 5:
        restart_count += 1
        rng = _random.Random(10000 + restart_count)
        cliques = _greedy_partition_fast(edges, n, rng)

        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = list(cliques)

        # Try local improvement on the best solution found so far
        if best_cliques is not None:
            improved = _local_improve(best_cliques, edges, n, _random.Random(20000 + restart_count))
            if len(improved) < best_count:
                best_count = len(improved)
                best_cliques = improved

        # Early stop if we reach a very good solution
        if best_count < 15:
            if restart_count > 100:
                break

    if best_cliques is None:
        # Fallback
        rng = _random.Random(0)
        best_cliques = _greedy_partition_fast(edges, n, rng)

    # Final validation
    if not _validate_partition(best_cliques, edges):
        # This should never happen; if it does, fall back to edge-by-edge
        best_cliques = [[u, v] for u, v in edges]

    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Solve smaller instance first
    res1 = _solve_instance(edges1, 32)

    # Solve larger instance with remaining time
    res2 = _solve_instance(edges2, 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
