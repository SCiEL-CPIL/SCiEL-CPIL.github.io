"""Sum-free subset maximizer: greedy + local search (add / remove / 2-opt swap)."""

import time
from random import Random

DEADLINE = 130.0  # wall-clock safety margin below the 150s kill


def construct():
    t0 = time.time()

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve(elements, deadline):
        n = len(elements)
        idx = {v: i for i, v in enumerate(elements)}
        # Precompute, for each element e, the list of pairs (i, j) with i<=j,
        # elements[i]+elements[j]==e.  These are the "sums" that e participates in.
        # For conflict counting we need: given current set S, an element x is
        # addable iff no pair (a,b) in S with a+b==x and x not already in S,
        # and no a in S with 2a==x.
        # Represent conflicts via: for each element x, list of "sum-events"
        # sum-events[x] = list of (a, b) such that a+b == x (a<=b, a,b in elements)
        sum_events = {i: [] for i in range(n)}
        for i in range(n):
            ai = elements[i]
            for j in range(i, n):
                s = ai + elements[j]
                if s in idx:
                    sum_events[idx[s]].append((i, j))

        def conflict_count(S):
            # number of violating triples (a,b,c) with a+b=c, a,b,c in S
            c = 0
            for i in S:
                for (a, b) in sum_events[i]:
                    if a in S and b in S:
                        c += 1
            return c

        # --- initial solution: all odds ---
        S = set(i for i in range(n) if elements[i] % 2 != 0)

        # --- greedy add: repeatedly add the element that fixes nothing and
        #     adds 1, i.e. is conflict-free.  We add any conflict-free element. ---
        def greedy_add(S):
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if i in S:
                        continue
                    ok = True
                    for (a, b) in sum_events[i]:
                        if a in S and b in S:
                            ok = False
                            break
                    if ok:
                        S.add(i)
                        changed = True
            return S

        S = greedy_add(S)
        best = S
        best_score = len(S)

        def valid(S):
            return conflict_count(S) == 0

        rng = Random(12345)

        def try_add(S, i):
            for (a, b) in sum_events[i]:
                if a in S and b in S:
                    return False
            return True

        # --- local search ---
        def local_search(S, deadline):
            S = set(S)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # 1) single add
                for i in range(n):
                    if i not in S and try_add(S, i):
                        S.add(i)
                        improved = True
                        break
                # 2) single remove to enable adds: remove r, then greedily add
                if time.time() >= deadline:
                    break
                for r in range(n):
                    if r not in S:
                        continue
                    cand = set(S)
                    cand.remove(r)
                    # greedily add what becomes possible
                    g = greedy_add(cand)
                    if len(g) > len(S):
                        S = g
                        improved = True
                        break
                # 3) 2-opt swap: remove up to 2, add up to 3
                if time.time() >= deadline:
                    break
                # small random perturbation to escape local optima
                if rng.random() < 0.15:
                    cand = set(S)
                    # remove 1-2 random present
                    present = list(cand)
                    k = rng.randint(1, 2)
                    for _ in range(min(k, len(present))):
                        r = rng.choice(present)
                        cand.remove(r)
                        if r in present:
                            present.remove(r)
                    g = greedy_add(cand)
                    if len(g) > len(S):
                        S = g
                        improved = True
                        break
            return S

        # iterate local search from best, with restarts
        while time.time() < deadline:
            S = local_search(best, deadline)
            if len(S) > best_score:
                best = S
                best_score = len(S)
            # random restart: start from odds + random evens
            if time.time() < deadline - 5:
                restart = set(i for i in range(n) if elements[i] % 2 != 0)
                # add random evens that are conflict-free
                evens = [i for i in range(n) if elements[i] % 2 == 0]
                rng.shuffle(evens)
                for i in evens:
                    if try_add(restart, i):
                        restart.add(i)
                S = local_search(restart, deadline)
                if len(S) > best_score:
                    best = S
                    best_score = len(S)

        return [elements[i] for i in best]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        # per-instance time budget
        remain = DEADLINE - (time.time() - t0)
        per = max(1.0, remain / 3.0)
        elems = instance(size, seed)
        results.append(solve(elems, t0 + per))
    return results
