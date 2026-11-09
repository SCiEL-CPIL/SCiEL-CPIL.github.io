"""
Clique partition of dense random graphs.

Strategy:
  1. Enumerate all maximal cliques (Bron-Kerbosch with pivot, bitmasks) once per instance.
  2. Greedily select edge-disjoint cliques from this pool, prioritising by a
     "value" score: edges covered / (1 + lambda * log(1 + shared-vertex count)).
     This encourages picking large cliques that don't share many vertices with
     already-chosen cliques, reducing fragmentation.
  3. Any remaining edges are covered as individual 2-vertex cliques.
  4. If time permits, a second pass with a different tie-breaking (randomised
     shuffle) is attempted and the better result is kept.
"""

import time
import itertools
from random import Random

_START = time.time()
_LIMIT = 115.0  # safety margin under 120 s wall-clock


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _random_edges(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _adj_mask(n, edges):
    m = [0] * n
    for (u, v) in edges:
        m[u] |= 1 << v
        m[v] |= 1 << u
    return m


def _iter_bits(x):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


def _maximal_cliques_bit(adj_mask, n, min_size=2, cap=200000):
    """Enumerate maximal cliques (as bitmasks) via Bron-Kerbosch with pivot."""
    result = []
    aborted = [False]

    def bronk(R, P, X):
        if aborted[0]:
            return
        if not _deadline_ok():
            aborted[0] = True
            return
        if not P and not X:
            if R.bit_count() >= min_size:
                result.append(R)
                if len(result) >= cap:
                    aborted[0] = True
            return
        if not P:
            return
        ux = P | X
        best_u = -1
        best_c = -1
        for u in _iter_bits(ux):
            c = (P & adj_mask[u]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u
        for v in _iter_bits(P & ~adj_mask[best_u]):
            bit = 1 << v
            bronk(R | bit, P & adj_mask[v], X & adj_mask[v])
            if aborted[0]:
                return
            P &= ~bit
            X |= bit

    bronk(0, (1 << n) - 1, 0)
    return result


def _mask_to_list(mask, n):
    return [v for v in range(n) if (mask >> v) & 1]


def _clique_edges(clique):
    return frozenset(_edge_key(u, v) for u, v in itertools.combinations(clique, 2))


def _solve_instance(n, p, seed, rng_seed=None):
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    if not all_edges:
        return []

    adj = _adj_mask(n, edges)

    # Enumerate maximal cliques of size >= 2
    masks = _maximal_cliques_bit(adj, n, min_size=2, cap=200000)
    cliques = [_mask_to_list(m, n) for m in masks]

    # Precompute edge sets and vertex sets for each clique
    clique_info = []
    for c in cliques:
        ec = _clique_edges(c)
        if not ec:
            continue
        vs = frozenset(c)
        clique_info.append((c, ec, vs))

    # Sort by size descending as a starting order
    clique_info.sort(key=lambda x: -len(x[0]))

    used_edges = set()
    chosen = []
    # Track how many times each vertex has been used in a chosen clique
    vertex_use = [0] * n

    # Greedy selection: repeatedly pick the clique with best value
    # value = |edges not yet used| / (1 + 0.5 * max_vertex_use_in_clique)
    # This favours large cliques and those using less-used vertices.
    remaining_info = list(range(len(clique_info)))

    while remaining_info and _deadline_ok():
        best_idx = -1
        best_val = -1
        for idx in remaining_info:
            c, ec, vs = clique_info[idx]
            unused = len(ec - used_edges)
            if unused == 0:
                continue
            # Penalty: max number of times any vertex in this clique has been used
            max_use = max(vertex_use[v] for v in vs)
            val = unused / (1.0 + 0.5 * max_use)
            if val > best_val:
                best_val = val
                best_idx = idx

        if best_idx == -1:
            break

        c, ec, vs = clique_info[best_idx]
        used_edges |= ec
        chosen.append(c)
        for v in vs:
            vertex_use[v] += 1
        remaining_info.remove(best_idx)

    # Cover remaining edges as individual 2-vertex cliques
    for e in all_edges - used_edges:
        chosen.append([e[0], e[1]])

    return chosen


def _solve_instance_randomized(n, p, seed):
    """Same as _solve_instance but with randomised tie-breaking among equal-value cliques."""
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    if not all_edges:
        return []

    adj = _adj_mask(n, edges)
    masks = _maximal_cliques_bit(adj, n, min_size=2, cap=200000)
    cliques = [_mask_to_list(m, n) for m in masks]

    clique_info = []
    for c in cliques:
        ec = _clique_edges(c)
        if not ec:
            continue
        vs = frozenset(c)
        clique_info.append((c, ec, vs))

    # Shuffle for randomised tie-breaking
    rng = Random(0x1234)
    rng.shuffle(clique_info)

    used_edges = set()
    chosen = []
    vertex_use = [0] * n
    remaining_info = list(range(len(clique_info)))

    while remaining_info and _deadline_ok():
        best_idx = -1
        best_val = -1
        for idx in remaining_info:
            c, ec, vs = clique_info[idx]
            unused = len(ec - used_edges)
            if unused == 0:
                continue
            max_use = max(vertex_use[v] for v in vs)
            val = unused / (1.0 + 0.5 * max_use)
            if val > best_val:
                best_val = val
                best_idx = idx

        if best_idx == -1:
            break

        c, ec, vs = clique_info[best_idx]
        used_edges |= ec
        chosen.append(c)
        for v in vs:
            vertex_use[v] += 1
        remaining_info.remove(best_idx)

    for e in all_edges - used_edges:
        chosen.append([e[0], e[1]])

    return chosen


def construct():
    res1 = _solve_instance(32, 0.6, 601)
    if not _deadline_ok():
        # Fallback: cover remaining as edges
        edges2 = _random_edges(64, 0.55, 602)
        res2 = [[e[0], e[1]] for e in edges2]
        return [res1, res2]

    res2 = _solve_instance(64, 0.55, 602)

    # If time permits, try randomised version and keep the better
    if _deadline_ok() and (time.time() - _START) < _LIMIT - 25.0:
        try:
            a1 = _solve_instance_randomized(32, 0.6, 601)
            if _deadline_ok() and (time.time() - _START) < _LIMIT - 10.0:
                a2 = _solve_instance_randomized(64, 0.55, 602)
                if len(a1) + len(a2) < len(res1) + len(res2):
                    res1, res2 = a1, a2
        except Exception:
            pass

    return [res1, res2]
