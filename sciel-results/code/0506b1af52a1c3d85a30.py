import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 150
    max_val = 10**18

    # First 16 primes (16th exceeds the 15-prime product)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    # ---- Fast score: number of adjacent coprime pairs in sorted divisors ----
    def tau_perp(n):
        if n < 2:
            return 0
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                cur = len(divs)
                factor = 1
                for _ in range(count):
                    factor *= p
                    for i in range(cur):
                        divs.append(divs[i] * factor)
        if temp > 1:
            cur = len(divs)
            for i in range(cur):
                divs.append(divs[i] * temp)
        divs.sort()
        score = 0
        m = len(divs)
        for i in range(m - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score

    def feasible(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
                if prod > max_val:
                    return False
        return True

    def score_exps(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
        return tau_perp(prod), prod

    # ---- Local search over exponent vectors ----
    def local_search(exps, iters=3000):
        cur_exps = list(exps)
        cur_score, cur_prod = score_exps(cur_exps)
        for _ in range(iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            improved = False
            for i in range(16):
                # increase
                if feasible(cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]):
                    new = cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]
                    s, pr = score_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score, cur_prod = new, s, pr
                        improved = True
                # decrease
                if cur_exps[i] > 0:
                    new = cur_exps[:i] + [cur_exps[i] - 1] + cur_exps[i + 1:]
                    s, pr = score_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score, cur_prod = new, s, pr
                        improved = True
            if not improved:
                break
        return cur_score, cur_prod, cur_exps

    # ---- Phase 1: square-free subsets (product of smallest primes) ----
    best_score = -1
    best_n = 2
    best_exps = [0] * 16

    # Product of first k primes, k = 0..15
    prod = 1
    for k in range(16):
        if k < 15:
            prod *= primes[k]
        if prod > max_val:
            break
        exps = [1] * (k + 1) + [0] * (15 - k)
        s = tau_perp(prod)
        if s > best_score:
            best_score, best_n, best_exps = s, prod, exps

    # ---- Phase 2: refine each near-optimal square-free vector ----
    for k in range(13, 16):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        base = [1] * (k + 1) + [0] * (15 - k)
        if not feasible(base):
            continue
        s, pr, _ = local_search(base, iters=2000)
        if s > best_score:
            best_score, best_n, best_exps = s, pr, _

    # ---- Phase 3: randomized restarts ----
    rng = random.Random(12345)
    for restart in range(20):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        exps = [0] * 16
        # random distinct-prime base
        for i in range(16):
            if rng.random() < 0.5:
                exps[i] = 1
        if not feasible(exps):
            continue
        s, pr, _ = local_search(exps, iters=1500)
        if s > best_score:
            best_score, best_n, best_exps = s, pr, _

    return best_n
