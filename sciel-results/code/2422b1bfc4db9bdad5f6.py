"""Clique partition via batched maximal-clique set-packing with time budget."""
# EVOLVE-BLOCK-START
import time
import itertools
import heapq

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


def _largest_clique_containing_edge(adj_mask, n, u, v, deadline_ok=None):
    """
    Find the largest maximal clique containing edge (u, v).
    """
    if deadline_ok and not deadline_ok():
        return None
    
    P = adj_mask[u] & adj_mask[v]
    
    best_clique_mask = 0
    best_size = 0
    
    def bronk(R, P_local, X):
        nonlocal best_clique_mask, best_size
        if deadline_ok and not deadline_ok():
            return
        if not P_local:
            if not X:
                size = R.bit_count()
                if size > best_size:
                    best_size = size
                    best_clique_mask = R
            return
        
        ux = P_local | X
        best_u = -1
        best_c = -1
        for u_p in _iter_bits(ux):
            c = (P_local & adj_mask[u_p]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u_p
                
        if best_u >= 0:
            P_sub = P_local & ~adj_mask[best_u]
        else:
            P_sub = P_local
            
        for v in _iter_bits(P_sub):
            bit = 1 << v
            bronk(R | bit, P_local & adj_mask[v], X & adj_mask[v])
            P_local &= ~bit
            X |= bit
            
    bronk(0, P, 0)
    
    if best_clique_mask == 0:
        return [u, v]
    
    verts = [u, v]
    for x in _iter_bits(best_clique_mask):
        verts.append(x)
    return verts

def _solve_instance(n, p, seed, per_pass_budget=6.0):
    """
    Partition edges into minimum number of cliques.
    Greedy: pick edge with max common neighbors, find largest maximal clique containing it.
    """
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    remaining = set(all_edges)

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

        # Find the "best" edge: max common neighbors
        best_edge = None
        best_common = -1
        
        for e in remaining:
            u, v = e
            common = rem_mask[u] & rem_mask[v]
            c = common.bit_count()
            if c > best_common:
                best_common = c
                best_edge = (u, v)
                
        if not best_edge:
            break
            
        u, v = best_edge
        
        # Find largest maximal clique containing this edge
        clique = _largest_clique_containing_edge(rem_mask, n, u, v, _deadline_ok)
        
        if not clique:
            clique = [u, v]
            
        chosen.append(clique)
        
        # Remove edges in this clique from remaining
        clique_edges = _clique_edges(clique)
        remaining -= clique_edges
        
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
