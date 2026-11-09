"""Small inclusion-maximal Sidon set in [1,256] via randomized construction + local search."""
import time
import random

N = 256
FULL = set(range(1, N + 1))


def _sums_of(S):
    s = set()
    L = list(S)
    for i in range(len(L)):
        a = L[i]
        s.add(2 * a)
        for j in range(i + 1, len(L)):
            s.add(a + L[j])
    return s


def _is_sidon(S):
    return len(_sums_of(S)) == len(S) * (len(S) + 1) // 2


def _is_maximal(S):
    Sset = set(S)
    if not Sset.issubset(FULL):
        return False
    sums = _sums_of(S)
    for x in FULL - Sset:
        if all((x + a) not in sums for a in S) and (2 * x) not in sums:
            return False
    return True


def _is_valid(S):
    Sset = set(S)
    if not Sset or len(Sset) != len(S):
        return False
    if not Sset.issubset(FULL):
        return False
    return _is_sidon(Sset) and _is_maximal(Sset)


def _greedy_construct(order):
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected


def _local_search(S, deadline, rng):
    Sset = set(S)
    if not _is_sidon(Sset) or not _is_maximal(Sset):
        return Sset
    best = Sset
    best_size = len(best)
    while time.time() < deadline:
        improved = False
        # try removals
        for a in list(best):
            cand = best - {a}
            if len(cand) >= best_size - 1 and _is_sidon(cand) and _is_maximal(cand):
                best = cand
                best_size = len(best)
                improved = True
                break
        if improved:
            continue
        # try swaps
        done = False
        for a in list(best):
            if done:
                break
            for b in FULL - best:
                cand = (best - {a}) | {b}
                if _is_sidon(cand) and _is_maximal(cand):
                    best = cand
                    best_size = len(best)
                    improved = True
                    done = True
                    break
        if improved:
            continue
        # try 2-for-2 swaps
        improved = False
        done = False
        bl = list(best)
        for i in range(len(bl)):
            if done:
                break
            for j in range(i + 1, len(bl)):
                a1, a2 = bl[i], bl[j]
                for b1 in FULL - best:
                    if done:
                        break
                    for b2 in FULL - best:
                        if b1 == b2:
                            continue
                        cand = (best - {a1, a2}) | {b1, b2}
                        if _is_sidon(cand) and _is_maximal(cand):
                            best = cand
                            best_size = len(best)
                            improved = True
                            done = True
                            break
        if improved:
            continue
        # no improvement: random perturbation to escape local optimum
        if time.time() > deadline - 0.05:
            break
        k = rng.choice([1, 2, 3])
        S = list(best)
        for _ in range(k):
            if not S:
                break
            a = S.pop(rng.randrange(len(S)))
            while True:
                b = rng.randrange(1, N + 1)
                if b not in S:
                    S.append(b)
                    break
        Sset = set(S)
        if _is_sidon(Sset) and _is_maximal(Sset):
            if len(Sset) < best_size:
                best = Sset
                best_size = len(Sset)
        # otherwise keep best
    return best


def construct():
    deadline = time.time() + 50.0
    best = None
    best_size = 10**9
    rng = random.Random(12345)

    # Phase 1: greedy with several random orders, keep smallest valid
    orders = [list(range(1, N + 1))]
    for _ in range(20):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)
    for o in orders:
        S = _greedy_construct(o)
        Sset = set(S)
        if _is_sidon(Sset) and _is_maximal(Sset):
            if len(Sset) < best_size:
                best = Sset
                best_size = len(Sset)
        if time.time() > deadline - 10.0:
            break

    # Phase 2: local search from the best seed
    if best is not None:
        best = _local_search(list(best), deadline, rng)
        best_size = len(best)

    # Phase 3: more randomized restarts + local search if time remains
    restarts = 0
    while time.time() < deadline - 2.0:
        o = list(range(1, N + 1))
        rng.shuffle(o)
        S = _greedy_construct(o)
        Sset = set(S)
        if _is_sidon(Sset) and _is_maximal(Sset):
            if len(Sset) < best_size:
                best = Sset
                best_size = len(Sset)
            # local search on this seed
            if time.time() < deadline - 2.0:
                cand = _local_search(list(Sset), min(deadline - 2.0, time.time() + 5.0), rng)
                if len(cand) < best_size:
                    best = cand
                    best_size = len(cand)
        restarts += 1
        if restarts > 400:
            break

    if best is None or len(best) == 0:
        # fallback: a guaranteed valid (large) maximal Sidon set
        S = _greedy_construct(list(range(1, N + 1)))
        return S
    return list(best)
