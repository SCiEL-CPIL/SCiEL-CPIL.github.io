"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

N, M, MAXV = 14, 35, 10000
ALL_EDGES = list(combinations(range(N), 2))


def _score_of(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _opt_edges(values, edges0, time_limit):
    """Minimize score by single-edge swaps; O(1) per move via counters."""
    if time_limit <= 0:
        return _score_of(values, edges0), list(edges0)
    edge_set = set(frozenset(e) for e in edges0)
    es, ep = {}, {}
    for i, j in ALL_EDGES:
        a, b = values[i], values[j]
        key = frozenset((i, j))
        es[key] = a + b
        ep[key] = a * b
    sc = Counter()
    pc = Counter()
    for fs in edge_set:
        sc[es[fs]] += 1
        pc[ep[fs]] += 1

    def cur():
        return max(sum(1 for c in sc.values() if c),
                   sum(1 for c in pc.values() if c))

    best = cur()
    best_set = set(edge_set)
    all_fs = [frozenset(e) for e in ALL_EDGES]
    deadline = time.time() + time_limit
    while time.time() < deadline:
        in_e = random.choice(list(edge_set))
        out_e = random.choice(all_fs)
        if out_e == in_e or out_e in edge_set:
            continue
        si, pi = es[in_e], ep[in_e]
        so, po = es[out_e], ep[out_e]
        sc[si] -= 1; pc[pi] -= 1
        sc[so] += 1; pc[po] += 1
        new = cur()
        if new < best:
            edge_set.remove(in_e)
            edge_set.add(out_e)
            best = new
            best_set = set(edge_set)
        else:
            sc[si] += 1; pc[pi] += 1
            sc[so] -= 1; pc[po] -= 1
    return best, [tuple(sorted(e)) for e in best_set]


def _divisors_of_h(h, count):
    ds = sorted(d for d in range(1, h + 1) if h % d == 0)
    if len(ds) >= count:
        return ds[:count]
    return None


def construct():
    t0 = time.time()
    TOTAL = 85.0
    random.seed(20240517)
    best = (10 ** 9, None, None)

    def consider(vals, edges):
        nonlocal best
        s = _score_of(vals, edges)
        if s < best[0]:
            best = (s, list(vals), list(edges))

    # ---- candidate value families ----
    candidates = []
    candidates.append(list(range(1, N + 1)))
    candidates.append([2 ** i for i in range(N)])
    for base in range(1, 9):
        cand = list(range(base, base + 2 * N, 2))[:N]
        if len(cand) == N and len(set(cand)) == N:
            candidates.append(cand)
    # divisors of highly composite numbers -> multiplicative collisions
    for h in (2520, 27720, 5040, 7560):
        d = _divisors_of_h(h, N)
        if d:
            candidates.append(d)
    # random dense small sets
    for _ in range(30):
        candidates.append(random.sample(range(1, 55), N))
    for _ in range(15):
        candidates.append(random.sample(range(1, 200), N))

    for vals in candidates:
        if time.time() - t0 > TOTAL - 6:
            break
        if len(set(vals)) != N or any(x < 1 or x > MAXV for x in vals):
            continue
        e0 = random.sample(ALL_EDGES, M)
        s, eo = _opt_edges(vals, e0, min(1.4, TOTAL - (time.time() - t0) - 6))
        consider(vals, eo)

    # ---- value perturbation around the best ----
    if best[1] is not None and time.time() - t0 < TOTAL - 6:
        bv = list(best[1])
        be = list(best[2])
        bs = best[0]
        deadline = t0 + TOTAL
        while time.time() < deadline:
            nv = bv[:]
            idx = random.randrange(N)
            nv[idx] = random.randint(1, MAXV)
            if len(set(nv)) != N:
                continue
            s, ne = _opt_edges(nv, be, min(0.7, deadline - time.time()))
            if s < bs:
                bv, be, bs = nv, ne, s
                consider(bv, be)

    vals = [int(x) for x in best[1]]
    edges = [[int(i), int(j)] for i, j in best[2]]

    # ---- validation (any failure -> fall back to a safe valid answer) ----
    ok = (len(vals) == N and len(set(vals)) == N and
          all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= MAXV for x in vals) and
          len(edges) == M)
    if ok:
        seen = set()
        for e in edges:
            if len(e) != 2 or e[0] == e[1] or not (0 <= e[0] < N and 0 <= e[1] < N):
                ok = False
                break
            fs = frozenset(e)
            if fs in seen:
                ok = False
                break
            seen.add(fs)
    if not ok:
        vals = list(range(1, N + 1))
        edges = [list(e) for e in ALL_EDGES[:M]]

    return {'values': vals, 'edges': edges}
# EVOLVE-BLOCK-END
