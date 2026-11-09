"""Combined: Bron-Kerbosch maximal cliques on remaining graph + greedy edge-cover refinement."""
# EVOLVE-BLOCK-START
import time
import itertools

_START = time.time()
_LIMIT = 115.0  # safety margin under 120s wall-clock


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _random_edges(n, p, seed):
    # Reproduce verifier's random_graph exactly: iterate u<v in order, rng.random()<p
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


def _maximal_cliques_bit(adj_mask, n, min_size=3, cap=300000, deadline_ok=None):
    """Enumerate maximal cliques (as bitmasks) via Bron-Kerbosch with pivot."""
    result = []
    aborted = [False]

    def bronk(R, P, X):
        if aborted[0]:
            return
        if deadline_ok is not None and not deadline_ok():
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
    chosen = []
    used = set()
    remaining = set(all_edges)

    rng = None
    if rng_seed is not None:
        from random import Random
        rng = Random(rng_seed)

    round_i = 0
    while remaining and _deadline_ok() and round_i < 50:
        round_i += 1
        rem_mask = _adj_mask(n, remaining)
        maxcls = _maximal_cliques_bit(rem_mask, n, 3, 300000, _deadline_ok)
        candidates = [_mask_to_list(m, n) for m in maxcls]
        if not candidates:
            break

        # Filter: keep only cliques whose all edges are still in remaining
        valid = []
        for c in candidates:
            ec = _clique_edges(c)
            if ec <= remaining:
                valid.append((c, ec))
        if not valid:
            break

        # Greedy: pick clique covering most remaining edges, tie-break by size, then rng
        if rng is not None:
            valid.sort(key=lambda x: (-len(x[1]), -len(x[0]), rng.random()))
        else:
            valid.sort(key=lambda x: (-len(x[1]), -len(x[0])))

        best_cl, best_ec = valid[0]
        chosen.append(best_cl)
        used |= best_ec
        remaining -= best_ec

    # Fallback: leftover edges as single-edge cliques
    for e in remaining:
        chosen.append([e[0], e[1]])
    return chosen


def construct():
    res1 = _solve_instance(32, 0.6, 601)
    res2 = _solve_instance(64, 0.55, 602)
    total = len(res1) + len(res2)

    # If time remains, try a randomized tie-break restart and keep the better.
    if _deadline_ok() and (time.time() - _START) < _LIMIT - 25.0:
        try:
            a1 = _solve_instance(32, 0.6, 601, rng_seed=0xDEADBEEF)
            if _deadline_ok() and (time.time() - _START) < _LIMIT - 10.0:
                a2 = _solve_instance(64, 0.55, 602, rng_seed=0xCAFEBABE)
                if len(a1) + len(a2) < total:
                    res1, res2 = a1, a2
        except Exception:
            pass

    return [res1, res2]
# EVOLVE-BLOCK-END
