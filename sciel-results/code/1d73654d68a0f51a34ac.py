"""Clique partition via batched maximal-clique set-packing with time budget."""
# EVOLVE-BLOCK-START
import time
import itertools
import heapq
import random

_START = time.time()
_LIMIT = 145.0  # safety margin under 150s wall-clock


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _elapsed():
    return time.time() - _START


def _random_edges(n, p, seed):
    """Reproduce verifier's random_graph: iterate u<v in order, rng.random()<p."""
    from random import Random
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


def _maximal_cliques_bit(adj_mask, n, min_size=2, deadline_ok=None, time_budget=None):
    """
    Enumerate maximal cliques (as bitmasks) via Bron-Kerbosch with pivot.
    Returns cliques with size >= min_size.
    Stops if deadline exceeded or time_budget (seconds) exhausted.
    """
    result = []
    aborted = [False]
    t0 = time.time()

    def bronk(R, P, X):
        if aborted[0]:
            return
        if deadline_ok is not None and not deadline_ok():
            aborted[0] = True
            return
        if time_budget is not None and (time.time() - t0) > time_budget:
            aborted[0] = True
            return
        if not P:
            if not X and R.bit_count() >= min_size:
                result.append(R)
            return

        # Pivot selection: pick u in P|X maximizing |P ∩ N(u)|
        ux = P | X
        best_u = -1
        best_c = -1
        for u in _iter_bits(ux):
            c = (P & adj_mask[u]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u

        if best_u >= 0:
            P_sub = P & ~adj_mask[best_u]
        else:
            P_sub = P

        for v in _iter_bits(P_sub):
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


def _solve_instance(n, p, seed, per_pass_budget=6.0):
    """
    Partition edges into minimum number of cliques.
    Each pass: find all maximal cliques of residual, greedily select a
    maximal set of edge-disjoint cliques (largest first), remove them.
    """
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    remaining = set(all_edges)

    # Cap on number of passes to avoid infinite loops
    max_passes = 200

    for _pass in range(max_passes):
        if not remaining:
            break
        if not _deadline_ok():
            break
        if _elapsed() > _LIMIT - 3.0:
            break

        # Build adjacency for remaining edges
        rem_mask = _adj_mask(n, remaining)

        # Find all maximal cliques in the remaining graph
        maxcls = _maximal_cliques_bit(
            rem_mask, n, min_size=2,
            deadline_ok=_deadline_ok,
            time_budget=per_pass_budget,
        )

        if not maxcls:
            break

        # Convert to (clique_list, edge_set) and filter to valid
        candidates = []
        for m in maxcls:
            c = _mask_to_list(m, n)
            if len(c) < 2:
                continue
            ec = _clique_edges(c)
            if ec <= remaining:
                candidates.append((c, ec))

        if not candidates:
            break

        # Sort by (edge_count, size) descending, then greedily pick edge-disjoint
        candidates.sort(key=lambda x: (len(x[1]), len(x[0])), reverse=True)

        used_edges = set()
        batch = []
        for c, ec in candidates:
            if ec & used_edges:
                continue
            batch.append(c)
            used_edges |= ec

        if not batch:
            # No progress; break to avoid infinite loop
            break

        chosen.extend(batch)
        remaining -= used_edges

    # Fallback: leftover edges as single-edge cliques
    for e in remaining:
        chosen.append([e[0], e[1]])

    return chosen


def _fallback_single_edges(n, p, seed):
    edges = _random_edges(n, p, seed)
    return [[u, v] for u, v in edges]


def construct():
    """
    Returns [cliques_for_instance1, cliques_for_instance2].
    """
    res1 = _solve_instance(32, 0.6, 601)
    if not _deadline_ok():
        res1 = _fallback_single_edges(32, 0.6, 601)

    res2 = _solve_instance(64, 0.55, 602)
    if not _deadline_ok():
        res2 = _fallback_single_edges(64, 0.55, 602)

    return [res1, res2]
# EVOLVE-BLOCK-END
