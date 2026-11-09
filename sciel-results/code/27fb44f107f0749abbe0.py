# EVOLVE-BLOCK-START
import time

def construct():
    DEADLINE = 110.0  # safety margin under 120s
    t0 = time.time()

    def remaining():
        return DEADLINE - (time.time() - t0)

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        # S: set of ints. Check no a+b=c with a,b,c in S (a=b allowed).
        lst = list(S)
        n = len(lst)
        # Build sumset via nested loop but early exit
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                b = lst[j]
                if a + b in S:
                    return False
        return True

    def greedy_add(S, cand_list, used):
        # Try to add elements from cand_list that keep sum-freeness.
        # used: set of indices already considered
        for idx in cand_list:
            if idx in used:
                continue
            v = S_list[idx]
            # Check adding v: need for all x in S: x+v not in S, and v+v not in S
            ok = True
            s2 = v + v
            if s2 in S:
                ok = False
            if ok:
                for x in S:
                    if (x + v) in S:
                        ok = False
                        break
            if ok:
                S.add(v)
                used.add(idx)

    def local_search(S, S_list, n, max_iters=2000):
        # S: set (mutable). S_list: original list.
        # 1) Try adding any element not in S that keeps sum-free.
        # 2) Try 2-opt: remove a, add b (b not in S) if keeps sum-free and |S| same (no size gain) -> skip.
        #    Actually we only add if size increases. 2-opt that keeps size is for diversity; skip.
        # We focus on additions; if stuck, try removing one element that blocks many adds, then re-add.
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding
            for i in range(n):
                v = S_list[i]
                if v in S:
                    continue
                ok = True
                if (v + v) in S:
                    ok = False
                if ok:
                    for x in list(S):
                        if (x + v) in S:
                            ok = False
                            break
                if ok:
                    S.add(v)
                    improved = True
            # Try remove-one-then-add-two: remove r in S, then try to add two new elements
            if not improved:
                # For each r in S, compute how many additions become possible
                best_gain = 0
                best_r = None
                best_adds = []
                for r in list(S):
                    S2 = set(S)
                    S2.discard(r)
                    adds = []
                    for i in range(n):
                        v = S_list[i]
                        if v in S2:
                            continue
                        ok = True
                        if (v + v) in S2:
                            ok = False
                        if ok:
                            for x in S2:
                                if (x + v) in S2:
                                    ok = False
                                    break
                        if ok:
                            adds.append(v)
                    gain = len(adds) - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                        best_adds = adds
                if best_gain > 0 and best_r is not None:
                    S.discard(best_r)
                    for v in best_adds:
                        S.add(v)
                    improved = True
        return S

    def solve_one(S_list, n, time_budget):
        # Start with odd numbers
        S = set(x for x in S_list if x % 2)
        # Local search
        S = local_search(S, S_list, n, max_iters=5000)
        # Multi-restart with random initial subsets if time permits
        best = S
        from random import Random
        rng = Random(12345)
        while time_budget() > 5:
            # Random initial subset: take a random 40% of elements, then grow
            init = set()
            for i in range(n):
                if rng.random() < 0.4:
                    init.add(S_list[i])
            # Greedy add
            used = set()
            for i in range(n):
                v = S_list[i]
                if v in init:
                    continue
                ok = True
                if (v + v) in init:
                    ok = False
                if ok:
                    for x in init:
                        if (x + v) in init:
                            ok = False
                            break
                if ok:
                    init.add(v)
            init = local_search(init, S_list, n, max_iters=2000)
            if len(init) > len(best):
                best = init
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S_list = instance(size, seed)
        n = len(S_list)
        # Allocate time proportionally to n, but ensure minimum
        # Budget: total ~100s; give more to larger n
        if size == 60:
            budget = 15.0
        elif size == 120:
            budget = 30.0
        else:
            budget = 60.0
        t_start = time.time()
        def tb():
            return budget - (time.time() - t_start)
        S = solve_one(S_list, n, tb)
        results.append(sorted(S))
        # Safety: if overall deadline near, break
        if remaining() < 10:
            # Fill remaining with odd numbers quickly
            for sz2, sd2 in ((60, 801), (120, 802), (240, 803)):
                if len(results) < 3:
                    sl = instance(sz2, sd2)
                    results.append(sorted(x for x in sl if x % 2))
            break
    return results
# EVOLVE-BLOCK-END
