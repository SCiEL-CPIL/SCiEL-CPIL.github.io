def construct():
    import random
    import time
    random.seed(12345)
    n = 40
    nums = list(range(2, n + 1))

    def can_add(S, x):
        for p in S:
            if p * x in S:
                return False
        return True

    def build_S(chosen):
        S = {1}
        for x in chosen:
            S = S | {p * x for p in S}
        return S

    def fill(chosen, S):
        chosen = list(chosen)
        chosen_set = set(chosen)
        changed = True
        while changed:
            changed = False
            for x in nums:
                if x in chosen_set:
                    continue
                if can_add(S, x):
                    S = S | {p * x for p in S}
                    chosen.append(x)
                    chosen_set.add(x)
                    changed = True
        return chosen, S

    def num_prime_factors(x):
        c = 0
        d = 2
        t = x
        while d * d <= t:
            while t % d == 0:
                c += 1
                t //= d
            d += 1
        if t > 1:
            c += 1
        return c

    primes = [x for x in nums if num_prime_factors(x) == 1]
    rest = [x for x in nums if x not in primes]

    orderings = []
    orderings.append(sorted(nums))
    orderings.append(sorted(nums, reverse=True))
    orderings.append(sorted(nums, key=lambda x: (num_prime_factors(x), x)))
    orderings.append(sorted(nums, key=lambda x: (-num_prime_factors(x), x)))
    orderings.append(primes + rest)
    orderings.append(primes + sorted(rest, key=lambda x: (num_prime_factors(x), x)))
    orderings.append(primes + sorted(rest, key=lambda x: (-num_prime_factors(x), x)))
    orderings.append(primes + sorted(rest, reverse=True))
    for _ in range(4):
        o = list(nums)
        random.shuffle(o)
        orderings.append(o)

    best = []
    best_S = {1}
    for order in orderings:
        S = {1}
        chosen = []
        for x in order:
            if can_add(S, x):
                S = S | {p * x for p in S}
                chosen.append(x)
        chosen, S = fill(chosen, S)
        if len(chosen) > len(best):
            best = chosen
            best_S = S

    # bounded local search: try swaps that enable more additions
    t0 = time.time()
    attempts = 0
    improved = True
    while improved and attempts < 200 and (time.time() - t0) < 40:
        improved = False
        cur_len = len(best)
        for c in list(best):
            for u in nums:
                if u in best:
                    continue
                attempts += 1
                if attempts >= 200 or (time.time() - t0) > 40:
                    break
                new_chosen = [x for x in best if x != c]
                new_chosen.append(u)
                new_S = build_S(new_chosen)
                new_chosen, new_S = fill(new_chosen, new_S)
                if len(new_chosen) > cur_len:
                    best, best_S = new_chosen, new_S
                    cur_len = len(best)
                    improved = True
                    break
            if improved:
                break

    return list(best)
