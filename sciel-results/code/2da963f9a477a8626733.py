"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt
    from collections import defaultdict
    import bisect

    start_time = time.time()
    TIME_LIMIT = 110

    LIMIT_LARGE = 10**10
    LIMIT_SMALL = 10**8

    # 1. Generate powerful numbers up to LIMIT_LARGE
    max_k = isqrt(LIMIT_LARGE)
    
    # Sieve for prime factors
    prime_factors = [[] for _ in range(max_k + 1)]
    for i in range(2, max_k + 1):
        if not prime_factors[i]:
            for j in range(i, max_k + 1, i):
                prime_factors[j].append(i)
                
    pw_large = set()
    for k in range(1, max_k + 1):
        pf = prime_factors[k]
        divisors = [1]
        for p in pf:
            new_divs = []
            for d in divisors:
                new_divs.append(d * p)
            divisors.extend(new_divs)
            
        for s in divisors:
            val = k * k * s
            if val <= LIMIT_LARGE:
                pw_large.add(val)

    pw_large_list = sorted(pw_large)
    pw_small_list = [x for x in pw_large_list if x <= LIMIT_SMALL]
    
    # 2. Count sums
    counts = defaultdict(int)
    pw_large_local = pw_large_list
    pw_small_local = pw_small_list
    counts_local = counts
    
    for a in pw_small_local:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        start_idx = bisect.bisect_left(pw_large_local, a)
        for j in range(start_idx, len(pw_large_local)):
            b = pw_large_local[j]
            s = a + b
            counts_local[s] += 1

    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}

    best_N, best_count = max(counts.items(), key=lambda x: x[1])
    
    # 3. Retrieve pairs
    pairs = []
    pw_set_local = pw_large
    max_a = best_N // 2
    end_idx = bisect.bisect_right(pw_large_list, max_a)
    
    for i in range(end_idx):
        a = pw_large_list[i]
        b = best_N - a
        if b in pw_set_local:
            if a <= b:
                pairs.append([a, b])
                
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
