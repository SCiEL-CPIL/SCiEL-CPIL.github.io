import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin before 120s limit

    # Step 1: Generate all powerful numbers up to a limit
    # A powerful number n has all prime factors with exponent >= 2
    # So n = a^2 * b^3 where b is square-free
    # We need n <= N_max, and we want to search for N up to some bound
    
    # Increasing M increases K (number of powerfuls) and thus the search space.
    # K ~ M^(2/5). Pair counting is O(K^2).
    # For M = 2e9, K ~ 6000, K^2 ~ 36e6, which is feasible in ~10-20s.
    
    M = 2_000_000_000  # Limit for powerful numbers
    
    powerful = set()
    
    # For each square-free b, add b^3 * a^2 for all a such that b^3 * a^2 <= M
    # b^3 <= M => b <= M^(1/3) ≈ 1260 for M = 2e9
    
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    # Precompute primes up to sqrt(max_b) for square-free check?
    # max_b is ~1260, sqrt is ~35. Simple trial division is fast enough.
    
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
    powerful_set = powerful
    
    if time.time() > deadline:
        # Fallback: use a smaller set if generation was unexpectedly slow
        M2 = 500_000_000
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
    
    from collections import defaultdict
    counts = defaultdict(int)
    
    best_N = 0
    best_count = 0
    
    # O(K^2) pair counting
    for i in range(K):
        a = powerful_list[i]
        for j in range(i, K):
            b = powerful_list[j]
            N = a + b
            counts[N] += 1
            if counts[N] > best_count:
                best_count = counts[N]
                best_N = N
            
            # Optional: break early if N is too large? 
            # No, we want to find the global max, and larger N might have more pairs.
            # However, since powerfuls are sparse, large Ns have fewer pairs.
            # But we don't know where the peak is.
            # Given K ~ 6000, the full loop is ~18M iterations, which is fast.
    
    # Rebuild pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and a <= b:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
