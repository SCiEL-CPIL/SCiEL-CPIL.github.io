"""Clique partition of two fixed dense random graphs via maximum clique enumeration + greedy set cover."""
# EVOLVE-BLOCK-START
import time
import random as _random
from itertools import combinations

_T0 = time.time()
_DEADLINE = _T0 + 110.0  # Wall-clock guard


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


def _popcount(x):
    return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')


def _find_max_cliques_bitmask(adj, n, max_size, deadline):
    """
    Find all maximum (or near-maximum) cliques using bitmask Bron-Kerbosch with pivot.
    Returns list of cliques (as frozensets of vertices).
    """
    cliques_found = []
    best_size = 0

    def bron_kerbosch(R, P, X):
        nonlocal best_size
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) >= 3:
                cliques_found.append(frozenset(R))
                if len(R) > best_size:
                    best_size = len(R)
            return

        # Pivot selection: choose vertex in P ∪ X with max |P ∩ N(u)|
        pivot_union = P | X
        if pivot_union:
            # Find vertex with max common neighbors in P
            best_pivot = -1
            best_count = -1
            temp = pivot_union
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                cnt = _popcount(P & adj[v])
                if cnt > best_count:
                    best_count = cnt
                    best_pivot = v
                temp ^= lsb
            candidates = P & ~adj[best_pivot]
        else:
            candidates = P

        temp = candidates
        while temp:
            if time.time() > deadline:
                return
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            bron_kerbosch(R | (1 << v), P & adj[v], X & adj[v])
            P ^= lsb
            X |= (1 << v)
            temp ^= lsb

    bron_kerbosch(0, (1 << n) - 1, 0)

    # If we didn't find enough, lower the threshold
    if len(cliques_found) < 5:
        cliques_found.clear()
        best_size = 0

        def bron_kerbosch2(R, P, X):
            nonlocal best_size
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques_found.append(frozenset(R))
                return

            pivot_union = P | X
            if pivot_union:
                best_pivot = -1
                best_count = -1
                temp = pivot_union
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    cnt = _popcount(P & adj[v])
                    if cnt > best_count:
                        best_count = cnt
                        best_pivot = v
                    temp ^= lsb
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P

            temp = candidates
            while temp:
                if time.time() > deadline:
                    return
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                bron_kerbosch2(R | (1 << v), P & adj[v], X & adj[v])
                P ^= lsb
                X |= (1 << v)
                temp ^= lsb

        bron_kerbosch2(0, (1 << n) - 1, 0)

    return cliques_found


def _greedy_set_cover(edges, n, cliques_pool, deadline):
    """
    Greedy set cover: repeatedly pick the clique covering the most uncovered edges.
    """
    if not edges:
        return []

    edge_set = set(_encode_edge(u, v) for u, v in edges)
    uncovered = set(edge_set)
    cliques = []

    # Pre-compute edge sets for each candidate clique
    clique_edge_sets = []
    for c in cliques_pool:
        verts = list(c)
        edge_list = []
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                e = _encode_edge(verts[i], verts[j])
                if e in edge_set:
                    edge_list.append(e)
        if edge_list:
            clique_edge_sets.append((c, frozenset(edge_list)))

    if not clique_edge_sets:
        # Fallback: just use 2-cliques
        for u, v in edges:
            cliques.append([u, v])
        return cliques

    # Sort by size descending for initial preference
    clique_edge_sets.sort(key=lambda x: -len(x[1]))

    while uncovered and time.time() < deadline:
        best_idx = -1
        best_overlap = -1
        for idx, (c, es) in enumerate(clique_edge_sets):
            overlap = len(uncovered & es)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        if best_idx == -1 or best_overlap <= 0:
            break

        c, es = clique_edge_sets[best_idx]
        cliques.append(list(c))
        uncovered -= es

        # Remove used clique from pool
        clique_edge_sets[best_idx] = None

    # Fallback for any remaining edges
    if uncovered:
        # Try to form 2-cliques for remaining
        remaining_edges = []
        for e in uncovered:
            u, v = _decode_edge(e)
            remaining_edges.append((u, v))
        for u, v in remaining_edges:
            cliques.append([u, v])

    return cliques


def _improve_partition(cliques, edges, n, deadline):
    """
    Try to merge adjacent cliques or find better partitions via local search.
    This is a simple improvement heuristic.
    """
    if len(cliques) <= 1:
        return cliques

    # Try to merge pairs of cliques if their union is also a clique
    # and doesn't conflict with other cliques
    edge_set = set(_encode_edge(u, v) for u, v in edges)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_cliques = []
        used = [False] * len(cliques)

        for i in range(len(cliques)):
            if used[i]:
                continue
            for j in range(i + 1, len(cliques)):
                if used[j]:
                    continue
                # Check if union of cliques[i] and cliques[j] is a clique
                union = set(cliques[i]) | set(cliques[j])
                is_clique = True
                verts = list(union)
                for a in range(len(verts)):
                    for b in range(a + 1, len(verts)):
                        if not (adj[verts[a]] & (1 << verts[b])):
                            is_clique = False
                            break
                    if not is_clique:
                        break

                if is_clique:
                    # Check that no edge in the union is covered by another clique
                    union_edges = set()
                    for a in range(len(verts)):
                        for b in range(a + 1, len(verts)):
                            union_edges.add(_encode_edge(verts[a], verts[b]))

                    conflict = False
                    for k in range(len(cliques)):
                        if k == i or k == j or used[k]:
                            continue
                        c_verts = cliques[k]
                        for a in range(len(c_verts)):
                            for b in range(a + 1, len(c_verts)):
                                if _encode_edge(c_verts[a], c_verts[b]) in union_edges:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break

                    if not conflict:
                        new_cliques.append(list(union))
                        used[i] = True
                        used[j] = True
                        improved = True
                        break

            if not improved:
                new_cliques.append(cliques[i])

        if improved:
            cliques = new_cliques

    return cliques


def _solve_instance(edges, n, deadline):
    """
    Solve a single instance:
    1. Enumerate large cliques using Bron-Kerbosch.
    2. Use greedy set cover to select a partition.
    3. Try local improvement.
    """
    if not edges:
        return []

    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Phase 1: Find large cliques
    sub_deadline = min(deadline - 2, _T0 + 30)
    cliques_pool = _find_max_cliques_bitmask(adj, n, 10, sub_deadline)

    # If we found very few, add all 3-cliques as candidates
    if len(cliques_pool) < 10 and time.time() < deadline - 5:
        # Add 3-cliques
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                    temp = common
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        if w > v:
                            cliques_pool.append(frozenset([u, v, w]))
                        temp ^= lsb

    # Phase 2: Greedy set cover
    cover_deadline = min(deadline - 3, _T0 + 80)
    partition = _greedy_set_cover(edges, n, cliques_pool, cover_deadline)

    # Phase 3: Local improvement
    if time.time() < deadline - 2:
        partition = _improve_partition(partition, edges, n, deadline)

    return partition


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


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Allocate time: 40s for graph1, 65s for graph2
    deadline1 = _T0 + 40
    deadline2 = _T0 + 108

    res1 = _solve_instance(edges1, 32, deadline1)
    res2 = _solve_instance(edges2, 64, deadline2)

    # Final validation
    if not _validate_partition(res1, edges1):
        # Fallback: 2-cliques
        res1 = [[u, v] for u, v in edges1]

    if not _validate_partition(res2, edges2):
        # Fallback: 2-cliques
        res2 = [[u, v] for u, v in edges2]

    return [res1, res2]
# EVOLVE-BLOCK-END
