# EVOLVE-BLOCK-START
import time
import itertools

_START = time.time()
_LIMIT = 110.0  # safety margin under 120s wall-clock


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


def _greedy_clique_bit(adj_mask, n):
    order = sorted(range(n), key=lambda x: -(adj_mask[x].bit_count()))
    cand = (1 << n) - 1
    clique = []
    for v in order:
        if not ((cand >> v) & 1):
            continue
        clique.append(v)
        cand &= adj_mask[v]
        if not cand:
            break
    return clique


def _solve_instance(n, p, seed, shuffle=False):
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    used = set()
    remaining = set(all_edges)
    rng2 = None
    if shuffle:
        from random import Random
        rng2 = Random(seed ^ 0x5DEECE66D)

    round_i = 0
    while remaining and _deadline_ok() and round_i < 40:
        round_i += 1
        rem_mask = _adj_mask(n, remaining)
        maxcls = _maximal_cliques_bit(rem_mask, n, 3, 300000, _deadline_ok)
        candidates = [_mask_to_list(m, n) for m in maxcls]
        if not candidates:
            c = _greedy_clique_bit(rem_mask, n)
            if len(c) >= 3:
                candidates = [c]
            else:
                break
        if shuffle and rng2 is not None:
            candidates.sort(key=lambda c: (-len(c), rng2.random()))
        else:
            candidates.sort(key=lambda c: -len(c))

        progress = False
        for c in candidates:
            if not _deadline_ok():
                break
            ec = _clique_edges(c)
            if ec & used:
                continue
            if not ec <= remaining:
                continue
            used |= ec
            remaining -= ec
            chosen.append(c)
            progress = True
        if not progress:
            break

    for e in remaining:
        chosen.append([e[0], e[1]])
    return chosen


def construct():
    res1 = _solve_instance(32, 0.6, 601)
    res2 = _solve_instance(64, 0.55, 602)

    # If time remains, try a randomized tie-break restart and keep the better.
    if _deadline_ok() and (time.time() - _START) < _LIMIT - 30.0:
        try:
            a1 = _solve_instance(32, 0.6, 601, shuffle=True)
            if _deadline_ok():
                a2 = _solve_instance(64, 0.55, 602, shuffle=True)
                if len(a1) + len(a2) < len(res1) + len(res2):
                    res1, res2 = a1, a2
        except Exception:
            pass

    return [res1, res2]
# EVOLVE-BLOCK-END
