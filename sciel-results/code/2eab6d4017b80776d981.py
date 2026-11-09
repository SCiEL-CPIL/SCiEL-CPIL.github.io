"""Rewritten: 3-uniform hypergraph independent set via greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_N = 256
_NODES = list(range(2, _N + 1))
_NN = len(_NODES)


def _bad_triple(a, b, c):
    return (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0


def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True


def construct():
    deadline = time.time() + 50
    rng = random.Random(2024)

    # Precompute pair_conflict[x] = set of (u,v), u<v, such that _bad_triple(x,u,v)
    pair_conflict = {}
    for x in _NODES:
        pc = set()
        for i in range(_NN):
            u = _NODES[i]
            if u == x:
                continue
            for j in range(i + 1, _NN):
                v = _NODES[j]
                if v == x:
                    continue
                if _bad_triple(x, u, v):
                    pc.add((u, v) if u < v else (v, u))
        pair_conflict[x] = pc

    def can_add(S, x):
        for (u, v) in pair_conflict[x]:
            if u in S and v in S:
                return False
        return True

    def local_search(S):
        S = set(S)
        # add all free
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in _NODES:
                if x not in S and can_add(S, x):
                    S.add(x)
                    changed = True
        # swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in _NODES:
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break
                conf_pairs = [(u, v) for (u, v) in pair_conflict[x] if u in S and v in S]
                if not conf_pairs:
                    continue
                elems = set()
                for (u, v) in conf_pairs:
                    elems.add(u)
                    elems.add(v)
                for e in elems:
                    S2 = S - {e}
                    if can_add(S2, x):
                        S = S2 | {x}
                        improved = True
                        break
                if improved:
                    break
        return S

    primes = [v for v in _NODES if _is_prime(v)]
    best = set(primes)
    best_size = len(best)

    cand = local_search(set(primes))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)

    for it in range(4000):
        if time.time() > deadline:
            break
        order = list(_NODES)
        rng.shuffle(order)
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        S = local_search(S)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    result = sorted(best)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                assert not _bad_triple(result[i], result[j], result[k])
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in result)
    return list(result)
# EVOLVE-BLOCK-END
