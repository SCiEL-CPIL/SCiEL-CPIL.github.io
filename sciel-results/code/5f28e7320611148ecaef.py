import time
import random
from math import gcd
from fractions import Fraction

_DEADLINE = 75.0  # safety margin under the 90s kill
_start = None


def _now():
    global _start
    if _start is None:
        _start = time.time()
    return time.time() - _start


def _denom(n):
    # common denominator of {1/i : i in 1..n}
    d = 1
    for i in range(1, n + 1):
        d = d * i // gcd(d, i)
    return d


def _subset_sums_scaled(scaled_values):
    # scaled_values: list of ints (sum of 2^k * v_k over k in subset)
    vals = sorted(scaled_values)
    res = [0]
    for v in vals:
        res += [x + v for x in res]
    return res


def _is_dissociated_ints(scaled_values):
    ss = _subset_sums_scaled(scaled_values)
    return len(set(ss)) == len(ss)


def _check_candidate(cand, D):
    # cand: iterable of ints in 1..22; D: common denominator
    sv = [D // x for x in cand]
    return _is_dissociated_ints(sv), sv


def _greedy_grow(seed_set, D, rng, deadline):
    # seed_set: list of ints; return best dissociated set found by greedy grow + small local search
    sv = [D // x for x in seed_set]
    base_ss = _subset_sums_scaled(sv)
    base_set = set(base_ss)

    current = list(seed_set)
    for attempt in range(200000):
        if _now() > deadline:
            break
        # try to add a new element
        best_new = None
        best_score = -1
        for x in range(1, 23):
            if x in current:
                continue
            v = D // x
            # quick prune: new sums = base_ss + v
            new_sums = [s + v for s in base_ss]
            # check no collision within new_sums or with base
            combined = base_set
            ok = True
            for ns in new_sums:
                if ns in combined:
                    ok = False
                    break
                combined.add(ns)
            if not ok:
                continue
            # score: spread (max-min) of combined
            score = max(combined) - min(combined)
            if score > best_score:
                best_score = score
                best_new = x
        if best_new is None:
            break
        current.append(best_new)
        sv.append(D // best_new)
        base_ss = _subset_sums_scaled(sv)
        base_set = set(base_ss)

    # local search: try single-element swaps to enable growth
    improved = True
    while improved and _now() < deadline:
        improved = False
        cur_set = set(current)
        for keep in list(current):
            for add in range(1, 23):
                if add in cur_set:
                    continue
                cand = [x for x in current if x != keep] + [add]
                ok, _ = _check_candidate(cand, D)
                if ok and len(cand) == len(current):
                    # same size; prefer higher min gap? just accept if improves spread
                    sv2 = [D // x for x in cand]
                    ss2 = _subset_sums_scaled(sv2)
                    spread2 = max(ss2) - min(ss2)
                    sv0 = [D // x for x in current]
                    ss0 = _subset_sums_scaled(sv0)
                    spread0 = max(ss0) - min(ss0)
                    if spread2 > spread0:
                        current = cand
                        improved = True
                        break
            if improved:
                break
    return current


def _try_grow_one(current, D, rng):
    # try to add a single element that keeps dissociation
    cur_set = set(current)
    candidates = [x for x in range(1, 23) if x not in cur_set]
    rng.shuffle(candidates)
    sv = [D // x for x in current]
    base_ss = _subset_sums_scaled(sv)
    base_set = set(base_ss)
    for x in candidates:
        v = D // x
        new_sums = [s + v for s in base_ss]
        combined = set(base_set)
        ok = True
        for ns in new_sums:
            if ns in combined:
                ok = False
                break
            combined.add(ns)
        if ok:
            current = current + [x]
            return current
    return current


def _random_candidate(rng, D):
    # build a random set of size k by random permutation greedy
    pool = list(range(1, 23))
    rng.shuffle(pool)
    cur = []
    sv = []
    base_ss = [0]
    base_set = {0}
    for x in pool:
        v = D // x
        new_sums = [s + v for s in base_ss]
        combined = set(base_set)
        ok = True
        for ns in new_sums:
            if ns in combined:
                ok = False
                break
            combined.add(ns)
        if ok:
            cur.append(x)
            sv.append(v)
            base_ss = _subset_sums_scaled(sv)
            base_set = set(base_ss)
    return cur


def construct():
    global _start
    _start = time.time()
    D = _denom(22)
    rng = random.Random(12345)

    # baseline known-good 9-set
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    ok, _ = _check_candidate(best, D)
    if not ok:
        best = [1]

    deadline = _start + _DEADLINE

    # 1) greedy grow from several seeds
    seeds = [
        [1],
        [2],
        [3],
        [1, 2],
        [1, 3],
        [1, 5],
        [1, 7],
        [1, 2, 3],
        [1, 2, 5],
        [1, 3, 7],
        list(range(1, 10)),
        list(range(1, 11)),
        list(range(1, 12)),
        best,
    ]
    for s in seeds:
        if _now() > deadline - 2:
            break
        s2 = [x for x in s if 1 <= x <= 22]
        ok, _ = _check_candidate(s2, D)
        if not ok:
            # repair: drop offending elements greedily
            s2 = []
            for x in sorted(s):
                cand = s2 + [x]
                if _check_candidate(cand, D)[0]:
                    s2 = cand
        grown = _greedy_grow(s2, D, rng, deadline)
        if len(grown) > len(best):
            best = grown

    # 2) random restarts: random candidate + grow
    budget = deadline - _now()
    if budget > 1.0:
        n_iters = int(min(200, max(10, budget / 0.3)))
        for i in range(n_iters):
            if _now() > deadline - 1:
                break
            cand = _random_candidate(rng, D)
            grown = _greedy_grow(cand, D, rng, deadline)
            if len(grown) > len(best):
                best = grown

    # 3) final attempt to grow best by one more element
    if _now() < deadline - 1:
        grown = _try_grow_one(best, D, rng)
        if len(grown) > len(best):
            best = grown

    # final safety: if best invalid for any reason, fall back
    ok, _ = _check_candidate(best, D)
    if not ok:
        return [1]
    return best


# self-test (not called by evaluator)
if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    print("result:", sorted(r), "size:", len(r))
    print("time:", time.time() - t0)
