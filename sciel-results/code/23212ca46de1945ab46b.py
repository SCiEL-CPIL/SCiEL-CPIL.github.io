"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import bisect
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin from 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    LIMIT = 10**18
    
    # Generate primes up to 10^5
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(100000)
    
    powerful_numbers = set()
    powerful_numbers.add(1)
    
    def dfs(idx, current):
        if time_left() < 5:
            return
        if current > LIMIT:
            return
        if current > 1:
            powerful_numbers.add(current)
        
        for i in range(idx, len(primes)):
            p = primes[i]
            p2 = p * p
            if p2 > LIMIT // current:
                break
            val = current * p2
            if val > LIMIT:
                break
            dfs(i + 1, val)
            # Higher powers of p
            while val * p <= LIMIT:
                val *= p
                if val > LIMIT:
                    break
                dfs(i + 1, val)
    
    dfs(0, 1)
    
    pn = sorted(powerful_numbers)
    pn_set = set(pn)
    n_pn = len(pn)
    
    if time_left() < 10:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    def count_pairs(N):
        count = 0
        half = N // 2
        for a in pn:
            if a > half:
                break
            if (N - a) in pn_set:
                count += 1
        return count
    
    # Generate candidates
    candidates = set()
    
    # 2 * pn[i] for sampled i
    step = max(1, n_pn // 2000)
    for i in range(0, n_pn, step):
        val = 2 * pn[i]
        if 2 <= val <= LIMIT:
            candidates.add(val)
    
    # Random pair sums
    random.seed(42)
    for _ in range(5000):
        i = random.randint(0, n_pn - 1)
        j = random.randint(0, n_pn - 1)
        if i > j:
            i, j = j, i
        val = pn[i] + pn[j]
        if 2 <= val <= LIMIT:
            candidates.add(val)
    
    # Values near median
    mid = n_pn // 2
    for offset in range(-200, 201, 5):
        idx = mid + offset
        if 0 <= idx < n_pn:
            val = 2 * pn[idx]
            if 2 <= val <= LIMIT:
                candidates.add(val)
    
    candidate_list = list(candidates)
    
    # Heuristic score: density of powerful numbers near N/2
    def heuristic(N):
        half = N / 2.0
        lo = bisect.bisect_left(pn, half * 0.85)
        hi = bisect.bisect_right(pn, half * 1.15)
        return hi - lo
    
    candidate_list.sort(key=heuristic, reverse=True)
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    for N in candidate_list:
        if time_left() < 20:
            break
        if N < 2 or N > LIMIT:
            continue
        cnt = count_pairs(N)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = [[a, N - a] for a in pn if a <= N // 2 and (N - a) in pn_set]
    
    # If count is low, try more targeted search
    if best_count < 20 and time_left() > 30:
        # Try N = pn[i] + pn[j] for i,j in a focused range
        # Focus on the densest region
        # Find the densest region of pn
        # Sample windows
        window_size = max(100, n_pn // 100)
        for start in range(0, n_pn - window_size, window_size // 10):
            if time_left() < 20:
                break
            # Try N values from sums in this window
            for _ in range(100):
                if time_left() < 20:
                    break
                i = start + random.randint(0, window_size - 1)
                j = start + random.randint(0, window_size - 1)
                if i > j:
                    i, j = j, i
                N = pn[i] + pn[j]
                if N < 2 or N > LIMIT or N in best_pairs:
                    continue
                cnt = count_pairs(N)
                if cnt > best_count:
                    best_count = cnt
                    best_N = N
                    best_pairs = [[a, N - a] for a in pn if a <= N // 2 and (N - a) in pn_set]
    
    if not best_pairs:
        best_N = 2
        best_pairs = [[1, 1]]
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
