import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 140
    max_val = 10**18

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    nprimes = len(primes)

    # Precompute prime powers for quick feasibility check
    prime_powers = []
    for p in primes:
        powers = [1]
        for _ in range(60):
            powers.append(powers[-1] * p)
        prime_powers.append(powers)

    def feasible(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= prime_powers[i][e]
                if prod > max_val:
                    return False
        return True

    def score_from_exps(exps):
        # Quick feasibility
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= prime_powers[i][e]
                if prod > max_val:
                    return -1

        # Build divisors as list of (value, mask)
        divs = [(1, 0)]
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            mask_bit = 1 << i
            cur_len = len(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for j in range(cur_len):
                    divs.append((divs[j][0] * factor, divs[j][1] | mask_bit))

        # Sort by value using numpy
        if len(divs) < 2:
            return 0
        vals = np.array([d[0] for d in divs], dtype=np.int64)
        masks = np.array([d[1] for d in divs], dtype=np.int64)
        order = np.argsort(vals, kind='stable')
        sorted_masks = masks[order]
        adjacent_and = sorted_masks[:-1] & sorted_masks[1:]
        score = np.count_nonzero(adjacent_and == 0)
        return int(score)

    def get_n(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= prime_powers[i][e]
        return prod

    def local_search(exps, iters=8000):
        cur_exps = list(exps)
        cur_score = score_from_exps(cur_exps)
        if cur_score < 0:
            return -1, 2, cur_exps

        for _ in range(iters):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            improved = False

            # Try increasing/decreasing each exponent
            for i in range(nprimes):
                # Increase
                if cur_exps[i] < 50:
                    new = cur_exps[:]
                    new[i] += 1
                    if feasible(new):
                        s = score_from_exps(new)
                        if s > cur_score:
                            cur_exps, cur_score = new, s
                            improved = True
                # Decrease
                if cur_exps[i] > 0:
                    new = cur_exps[:]
                    new[i] -= 1
                    s = score_from_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score = new, s
                        improved = True

            # Replace prime: remove prime i, add prime j with same exponent
            for i in range(nprimes):
                if cur_exps[i] == 0:
                    continue
                for j in range(nprimes):
                    if cur_exps[j] > 0 or j == i:
                        continue
                    new = cur_exps[:]
                    new[i] = 0
                    new[j] = cur_exps[i]
                    if feasible(new):
                        s = score_from_exps(new)
                        if s > cur_score:
                            cur_exps, cur_score = new, s
                            improved = True

            if not improved:
                break
        return cur_score, get_n(cur_exps), cur_exps

    best_score = -1
    best_n = 2
    best_exps = [0] * nprimes

    # Generate strong seeds
    seeds = []

    # 1. Product of first k primes (square-free)
    for k in range(1, 16):
        exps = [0] * nprimes
        for i in range(k):
            exps[i] = 1
        if feasible(exps):
            seeds.append(exps)

    # 2. N15 with extra powers of 2
    base_15 = [1]*15 + [0]
    for extra_2 in range(1, 6):
        exps = base_15[:]
        exps[0] += extra_2
        if feasible(exps):
            seeds.append(exps)

    # 3. N14 with powers of 2 and 3
    base_14 = [1]*14 + [0, 0]
    for e2 in range(1, 8):
        for e3 in range(0, 6):
            exps = base_14[:]
            exps[0] += e2
            exps[1] += e3
            if feasible(exps):
                seeds.append(exps)

    # 4. N13 with more flexibility
    base_13 = [1]*13 + [0, 0, 0]
    for e2 in range(1, 6):
        for e3 in range(0, 5):
            for e5 in range(0, 4):
                exps = base_13[:]
                exps[0] += e2
                exps[1] += e3
                exps[2] += e5
                if feasible(exps):
                    seeds.append(exps)

    # 5. Randomized seeds with diverse prime sets
    rng = random.Random(42)
    for _ in range(80):
        exps = [0] * nprimes
        size = rng.randint(13, 15)
        # Randomly select which primes to use
        available = list(range(nprimes))
        rng.shuffle(available)
        selected = available[:size]
        for i in selected:
            exps[i] = 1
        # Add extra power to a small-index prime
        if size < nprimes:
            extra_idx = rng.choice([i for i in range(3) if exps[i] < 5])
            exps[extra_idx] += rng.randint(1, 3)
        if feasible(exps):
            seeds.append(exps)

    # Evaluate seeds
    seed_scores = []
    for s in seeds:
        sc = score_from_exps(s)
        if sc > 0:
            seed_scores.append((sc, s))

    seed_scores.sort(reverse=True)

    # Run local search on top seeds
    for sc, exps in seed_scores[:30]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        final_score, final_n, final_exps = local_search(exps, iters=6000)
        if final_score > best_score:
            best_score = final_score
            best_n = final_n
            best_exps = final_exps

    if best_n > 10**18 or best_n < 2:
        return 2

    return best_n
