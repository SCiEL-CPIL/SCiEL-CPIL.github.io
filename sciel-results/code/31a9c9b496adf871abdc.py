import time
import math
import random

def _smooth_set(limit, primes):
    s = set()
    for n in range(2, limit + 1):
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            s.add(n)
    return s

def _find_clique(cands, smooth, size, deadline):
    n = len(cands)
    adj = [set() for _ in range(n)]
    for i in range(n):
        ai = cands[i]
        for j in range(i + 1, n):
            if (ai + cands[j]) in smooth:
                adj[i].add(j)
                adj[j].add(i)
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    best = []
    def bt(current, pool):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) == size:
            best = list(current)
            return
        if len(pool) < size - len(current):
            return
        for k, idx in enumerate(pool):
            if time.time() > deadline:
                return
            if not all(idx in adj[c] for c in current):
                continue
            new_pool = [x for x in pool[k + 1:] if x in adj[idx]]
            current.append(idx)
            bt(current, new_pool)
            current.pop()
            if best:
                return
    bt([], order)
    return best

def _find_clique_randomized(cands, smooth, size, deadline, n_trials=20):
    n = len(cands)
    adj = [set() for _ in range(n)]
    for i in range(n):
        ai = cands[i]
        for j in range(i + 1, n):
            if (ai + cands[j]) in smooth:
                adj[i].add(j)
                adj[j].add(i)
    deg = [len(adj[i]) for i in range(n)]
    best = []
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        # Randomized ordering: mix degree-based with random
        if trial == 0:
            order = sorted(range(n), key=lambda i: deg[i], reverse=True)
        else:
            order = list(range(n))
            random.shuffle(order)
            # Bias towards high-degree nodes
            order.sort(key=lambda i: (deg[i], random.random()), reverse=True)
        
        best_local = []
        def bt(current, pool):
            nonlocal best_local
            if time.time() > deadline:
                return
            if len(current) == size:
                best_local = list(current)
                return
            if len(pool) < size - len(current):
                return
            for k, idx in enumerate(pool):
                if time.time() > deadline:
                    return
                if not all(idx in adj[c] for c in current):
                    continue
                new_pool = [x for x in pool[k + 1:] if x in adj[idx]]
                current.append(idx)
                bt(current, new_pool)
                current.pop()
                if best_local:
                    return
        bt([], order)
        if best_local and len(best_local) > len(best):
            best = best_local
        if best:
            break
    return best

def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            all_primes |= _factorize(nums[i] + nums[j])
    return len(all_primes)

def construct():
    t0 = time.time()
    deadline = t0 + 75.0
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    best_result = None
    best_score = float('inf')
    
    for ps in prime_sets:
        if time.time() > deadline - 8.0:
            break
        for N in [300, 600, 1200, 2400]:
            if time.time() > deadline - 8.0:
                break
            smooth = _smooth_set(2 * N, ps)
            cands = list(range(1, N + 1))
            res = _find_clique(cands, smooth, 12, deadline - 8.0)
            if res:
                out = [cands[i] for i in res]
                if len(out) == 12 and len(set(out)) == 12:
                    score = _count_distinct_primes(out)
                    if score < best_score:
                        best_score = score
                        best_result = out
                    if best_score <= len(ps):
                        return out
    
    if best_result:
        return best_result
    
    # Fallback: try randomized search with smaller prime sets
    for ps in prime_sets[:2]:
        if time.time() > deadline - 5.0:
            break
        N = 600
        smooth = _smooth_set(2 * N, ps)
        cands = list(range(1, N + 1))
        res = _find_clique_randomized(cands, smooth, 12, deadline - 5.0, n_trials=10)
        if res:
            out = [cands[i] for i in res]
            if len(out) == 12 and len(set(out)) == 12:
                return out
    
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
