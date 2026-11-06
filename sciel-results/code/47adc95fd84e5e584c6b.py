import math
import time
import itertools
import random

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    n = 12

    # SPF table up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors(x):
        if x <= 1:
            return frozenset()
        s = set()
        while x > 1:
            p = spf[x]
            s.add(p)
            while x % p == 0:
                x //= p
        return frozenset(s)

    pf = {s: prime_factors(s) for s in range(2, MAX_SUM + 1)}

    def count(arr):
        u = set()
        for i in range(n - 1):
            ai = arr[i]
            for j in range(i + 1, n):
                u |= pf[ai + arr[j]]
        return len(u)

    # Build k-smooth pools for various prime sets
    def smooth_pool(primes, limit=10000):
        pool = set([1])
        for p in primes:
            new = set()
            for x in list(pool):
                v = x
                while v <= limit:
                    new.add(v)
                    v *= p
            pool |= new
        return sorted(pool)

    pool_7 = smooth_pool([2, 3, 5, 7])       
    pool_5 = smooth_pool([2, 3, 5])           
    pool_11 = smooth_pool([2, 3, 5, 7, 11])

    # Candidate pools to search in
    candidate_pools = [pool_7, pool_5, pool_11]

    # For a given prime set P (as a frozenset), build smooth-sum set
    def build_smooth_sum_set(primes):
        P = frozenset(primes)
        ss = set()
        for s in range(2, MAX_SUM + 1):
            if pf[s] <= P:
                ss.add(s)
        return ss

    # Enhanced Local Search with Simulated Annealing
    def local_search_sa(start, pool, max_iter=2000):
        cur = list(start)
        existing = set(cur)
        cur_score = count(cur)
        best_cur = list(cur)
        best_score = cur_score
        
        # Simple SA parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        it = 0
        while it < max_iter and time.time() < deadline - 3.0:
            it += 1
            if T < T_min:
                T = T_min
            
            # Pick a random index to change
            idx = random.randint(0, n - 1)
            old_val = cur[idx]
            
            # Pick a random candidate from the pool
            # To ensure we explore, we pick a random valid candidate
            # We can restrict to candidates that are "close" or just random from pool
            # Given pool size is manageable (<1000 for 7-smooth), random is fine
            cand = pool[random.randint(0, min(len(pool) - 1, 1000))]
            
            if cand == old_val or cand in existing:
                # Try to find a different one
                for _ in range(10):
                    cand = pool[random.randint(0, min(len(pool) - 1, 1000))]
                    if cand != old_val and cand not in existing:
                        break
                else:
                    continue
            
            # Calculate new score
            cur[idx] = cand
            new_score = count(cur)
            
            delta = new_score - cur_score
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept move
                existing.discard(old_val)
                existing.add(cand)
                cur_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_cur = list(cur)
                    if best_score <= 3:
                        return best_cur, best_score
            else:
                # Reject move
                cur[idx] = old_val
            
            T *= cooling
            
        return best_cur, best_score

    # Generate many seeds
    seeds = []

    # Seed A: greedy 12-clique in graph where edge = sum in smooth set
    for primes in ([2, 3, 5, 7], [2, 3, 5], [2, 3, 5, 7, 11], [2, 3, 5, 7, 11, 13]):
        ss = build_smooth_sum_set(primes)
        pool = smooth_pool(primes)
        # Greedy clique
        clique = []
        for u in pool:
            if u > 10000:
                break
            ok = True
            for v in clique:
                if (u + v) not in ss:
                    ok = False
                    break
            if ok:
                clique.append(u)
            if len(clique) >= n:
                break
        if len(clique) >= n:
            seeds.append(clique[:n])

    # Seed B: multiples of LCM-ish
    for base in (6, 30, 210, 420):
        multiples = [base * i for i in range(1, 100) if base * i <= 10000]
        if len(multiples) >= n:
            seeds.append(multiples[:n])
            for off in (1, 2, 3, 5):
                shifted = [base * i + off for i in range(1, 100) if base * i + off <= 10000]
                if len(shifted) >= n:
                    seeds.append(shifted[:n])

    # Seed C: small 7-smooth
    seeds.append(pool_7[:n])
    seeds.append(pool_5[:n])

    # Seed D: all even small
    seeds.append([2 * i for i in range(1, n + 1)])
    seeds.append([2 * i + 1 for i in range(1, n + 1)])
    seeds.append(list(range(1, n + 1)))
    seeds.append([1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30])
    
    # Seed E: Random samples from smooth pools
    for _ in range(20):
        random.seed(int(time.time() * 1000) + _)
        p = pool_7
        s = random.sample(p, n)
        seeds.append(s)

    # Deduplicate & validate seeds
    valid_seeds = []
    seen = set()
    for s in seeds:
        if len(set(s)) != n or any(x > 10000 for x in s) or any(x <= 0 for x in s):
            continue
        key = tuple(sorted(s))
        if key in seen:
            continue
        seen.add(key)
        valid_seeds.append(list(s))

    best = None
    bs = 10 ** 9
    for s in valid_seeds:
        c = count(s)
        if c < bs:
            bs = c
            best = list(s)
        if bs <= 3:
            return best

    # Run SA from best seed and other promising seeds
    for pool in candidate_pools:
        c, sc = local_search_sa(best, pool, max_iter=500)
        if sc < bs:
            bs = sc
            best = c
        if bs <= 3:
            return best

    # Run SA from other valid seeds if time permits
    for s in valid_seeds:
        if time.time() > deadline - 10.0:
            break
        for pool in candidate_pools:
            if time.time() > deadline - 5.0:
                break
            c, sc = local_search_sa(s, pool, max_iter=100)
            if sc < bs:
                bs = sc
                best = c
            if bs <= 3:
                return best

    return list(best)
