"""Sum-free subset construction with time-bounded local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin under 150s

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        # Can x be added to S_set while preserving sum-free?
        for a in S_set:
            if (a + x) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def greedy_build(S_all, order):
        S = set()
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    def local_search(S, S_all, budget_fn, max_rounds=50):
        S_set = set(S)
        S_list = list(S)
        excluded = [x for x in S_all if x not in S_set]
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and budget_fn():
            improved = False
            rounds += 1
            # Try to add each excluded element
            random.shuffle(excluded)
            for x in excluded:
                if can_add(x, S_set):
                    S_set.add(x)
                    excluded.remove(x)
                    improved = True
            # Try swaps: remove one from S, add one from excluded
            if not improved and excluded:
                random.shuffle(S_list)
                random.shuffle(excluded)
                for rm in S_list:
                    if not budget_fn():
                        break
                    S_set.discard(rm)
                    added_any = False
                    for x in excluded:
                        if can_add(x, S_set):
                            S_set.add(x)
                            excluded.remove(x)
                            added_any = True
                            break
                    if not added_any:
                        S_set.add(rm)
                    else:
                        improved = True
                        S_list = list(S_set)
                        if not budget_fn():
                            break
        return S_set

    def solve_one(S_all, budget_fn):
        n = len(S_all)
        best = set()
        best_size = 0

        restarts = 300
        for restart in range(restarts):
            if not budget_fn():
                break
            r = random.Random(restart * 7919 + n)
            if restart % 7 == 0:
                order = sorted(S_all)
            elif restart % 7 == 1:
                order = sorted(S_all, reverse=True)
            elif restart % 7 == 2:
                order = list(S_all)
                r.shuffle(order)
            elif restart % 7 == 3:
                order = sorted(S_all, key=lambda x: -abs(x))
            elif restart % 7 == 4:
                order = sorted(S_all, key=lambda x: (x % 2 == 0, -abs(x)))
            elif restart % 7 == 5:
                order = sorted(S_all, key=lambda x: x % 3)
            else:
                order = sorted(S_all, key=lambda x: -x)

            S = greedy_build(S_all, order)
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)

            # Local search
            S = local_search(S, S_all, budget_fn, max_rounds=20)
            if len(S) > best_size:
                best = S
                best_size = len(S)

        return list(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S_all = instance(size, seed)

        def budget_fn():
            return (time.time() - t0) < DEADLINE

        result = solve_one(S_all, budget_fn)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
