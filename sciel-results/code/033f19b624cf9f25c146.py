import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin before 120s limit

    # Step 1: Generate all powerful numbers up to a limit
    # A powerful number n has all prime factors with exponent >= 2
    # So n = a^2 * b^3 where b is square-free
    # We need n <= N_max, and we want to search for N up to some bound
    
    # The key insight: we want to find N with many representations as sum of two powerfuls.
    # Expected count scales as N^(1/3), so larger N gives more pairs.
    # But we're limited by the number of powerfuls below N.
    # Number of powerfuls up to X is ~ c * X^(2/5) + ... (heuristic)
    
    # Let's generate powerfuls up to a reasonable limit, say 50 million
    # This should be fast and give us enough candidates.
    
    M = 50_000_000  # Limit for powerful numbers
    
    powerful = set()
    
    # For each square-free b, add b^3 * a^2 for all a such that b^3 * a^2 <= M
    # b^3 <= M => b <= M^(1/3) ≈ 368 for M = 50M
    
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sf = True
        temp = b
        d = 2
        while d * d <= temp:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt >= 2:
                is_sf = False
                break
            d += 1 if d == 2 else 2
        if not is_sf:
            continue
        
        b_cubed = b ** 3
        if b_cubed > M:
            break
        
        # a^2 <= M / b_cubed => a <= sqrt(M / b_cubed)
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
    
    powerful_list = sorted(powerful)
    powerful_set = powerful  # Already a set
    
    if time.time() > deadline:
        # Fallback: use a smaller set
        M2 = 5_000_000
        powerful2 = set()
        max_b2 = int(M2 ** (1.0 / 3.0)) + 2
        for b in range(1, max_b2 + 1):
            is_sf = True
            temp = b
            d = 2
            while d * d <= temp:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt += 1
                if cnt >= 2:
                    is_sf = False
                    break
                d += 1 if d == 2 else 2
            if not is_sf:
                continue
            b_cubed = b ** 3
            if b_cubed > M2:
                break
            max_a = isqrt(M2 // b_cubed)
            for a in range(1, max_a + 1):
                powerful2.add(a * a * b_cubed)
        powerful_list = sorted(powerful2)
        powerful_set = powerful2
    
    K = len(powerful_list)
    
    # Step 2: For each powerful number a, for each powerful number b >= a, 
    # compute N = a + b and count how many pairs sum to each N.
    # But K can be large (~100k+), so O(K^2) is too slow.
    
    # Alternative: For each candidate N in a range, count pairs.
    # But we don't know which N to check.
    
    # Better approach: Use the fact that we want to maximize the count.
    # For a given N, the number of representations is the number of pairs (a, b) 
    # with a <= b, a + b = N, both powerful.
    
    # Strategy: 
    # 1. Generate all powerfuls up to M
    # 2. For a range of N values, count representations
    # 3. Return the best N found
    
    # Since the number of powerfuls grows as ~M^(2/5), for M=50M we get ~50M^(0.4) ≈ 2700 powerfuls
    # Actually let me estimate: the count of powerfuls up to X is ~ c * X^(2/5)
    # For X = 50M = 5*10^7, X^(2/5) = (5*10^7)^0.4 ≈ (5)^0.4 * (10^7)^0.4 ≈ 1.9 * 10^2.8 ≈ 1.9 * 631 ≈ 1200
    # So K is around 1000-2000, making O(K^2) feasible (~4M operations).
    
    # Let's do the O(K^2) approach: for each pair (i, j) with i <= j, 
    # N = powerful_list[i] + powerful_list[j], count occurrences.
    
    from collections import defaultdict
    counts = defaultdict(int)
    
    # Only consider N up to 2*M (since both a and b are <= M)
    # And we want N to be in a reasonable range.
    
    # To keep it manageable, we'll only count N in a window around the median or upper range.
    # Actually, let's just count all and keep track of the best.
    
    # But storing counts for all possible N (up to 100M) is fine with a dict.
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for i in range(K):
        a = powerful_list[i]
        # b must be >= a, so j >= i
        for j in range(i, K):
            b = powerful_list[j]
            N = a + b
            if N > 2 * M:
                break
            counts[N] += 1
            if counts[N] > best_count:
                best_count = counts[N]
                best_N = N
                # Don't store pairs yet, we'll rebuild at the end
    
    # Rebuild pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and a <= b:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
