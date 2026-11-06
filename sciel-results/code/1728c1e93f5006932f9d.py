import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 140
    max_val = 10**18

    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    nprimes = len(primes)

    # Precompute prime powers
    prime_powers = []
    for p in primes:
        pw = [1]
        while pw[-1] * p <= max_val:
            pw.append(pw[-1] * p)
        prime_powers.append(pw)

    def feasible(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= prime_powers[i][e]
                if prod > max_val:
                    return False
        return True

    def score_from_exps(exps):
        # Build divisors with bitmask tracking which primes divide them
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
        
        divs.sort(key=lambda x: x[0])
        score = 0
        m = len(divs)
        for i in range(m - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    def get_n(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= prime_powers[i][e]
        return prod

    def local_search(exps, iters=3000):
        cur_exps = list(exps)
        cur_score = score_from_exps(cur_exps)
        if cur_score < 0:
            return -1, 2, cur_exps
            
        for _ in range(iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            improved = False
            
            # Try increasing/decreasing each exponent
            for i in range(nprimes):
                # Increase
                if cur_exps[i] + 1 < len(prime_powers[i]):
                    new = cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]
                    if feasible(new):
                        s = score_from_exps(new)
                        if s > cur_score:
                            cur_exps, cur_score = new, s
                            improved = True
                # Decrease
                if cur_exps[i] > 0:
                    new = cur_exps[:i] + [cur_exps[i] - 1] + cur_exps[i + 1:]
                    s = score_from_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score = new, s
                        improved = True
            
            # Try swap moves: exchange 1 unit between two primes
            for i in range(nprimes):
                if cur_exps[i] == 0:
                    continue
                for j in range(i + 1, nprimes):
                    # Move one unit from i to j
                    if cur_exps[j] + 1 < len(prime_powers[j]):
                        new = cur_exps[:]
                        new[i] -= 1
                        new[j] += 1
                        if feasible(new):
                            s = score_from_exps(new)
                            if s > cur_score:
                                cur_exps, cur_score = new, s
                                improved = True
                    # Move one unit from j to i
                    if cur_exps[i] + 1 < len(prime_powers[i]) and cur_exps[j] > 0:
                        new = cur_exps[:]
                        new[j] -= 1
                        new[i] += 1
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

    # ---- Generate diverse seeds ----
    seeds = []
    
    # 1. Product of first k primes (primorials)
    for k in range(1, 16):
        exps = [0] * nprimes
        for i in range(k):
            exps[i] = 1
        if feasible(exps):
            seeds.append(exps)
    
    # 2. Primorial with extra powers of small primes
    for k in range(13, 16):
        base = [0] * nprimes
        for i in range(k):
            base[i] = 1
        for e2 in range(0, 5):
            for e3 in range(0, 5):
                exps = base[:]
                exps[0] += e2
                if e3 > 0:
                    exps[1] += e3
                if feasible(exps):
                    seeds.append(exps)
    
    # 3. Random square-free subsets
    rng = random.Random(42)
    for _ in range(30):
        exps = [0] * nprimes
        indices = list(range(15))
        rng.shuffle(indices)
        size = rng.randint(12, 15)
        for i in range(size):
            exps[indices[i]] = 1
        if feasible(exps):
            seeds.append(exps)
    
    # 4. Random with small exponents
    for _ in range(30):
        exps = [0] * nprimes
        for i in range(15):
            if rng.random() < 0.6:
                exps[i] = rng.randint(1, 3)
        if feasible(exps):
            seeds.append(exps)
    
    # Evaluate seeds and keep top ones
    seed_scores = []
    for s in seeds:
        sc = score_from_exps(s)
        if sc > 0:
            seed_scores.append((sc, s))
    
    seed_scores.sort(reverse=True)
    
    # Run local search on top seeds
    for sc, exps in seed_scores[:15]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        final_score, final_n, final_exps = local_search(exps, iters=2000)
        if final_score > best_score:
            best_score = final_score
            best_n = final_n
            best_exps = final_exps
    
    # Additional randomized restarts if time permits
    rng2 = random.Random(12345)
    for restart in range(10):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        exps = [0] * nprimes
        for i in range(15):
            if rng2.random() < 0.5:
                exps[i] = rng2.randint(1, 2)
        if not feasible(exps):
            continue
        s, pr, _ = local_search(exps, iters=1000)
        if s > best_score:
            best_score = s
            best_n = pr
            best_exps = _
    
    if best_n > max_val or best_n < 2:
        return 2
        
    return best_n
