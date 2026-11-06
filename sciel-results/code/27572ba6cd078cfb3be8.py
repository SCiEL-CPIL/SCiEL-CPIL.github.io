import time
import numpy as np
from math import isqrt

def is_powerful(n):
    """Check if n is powerful (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp == 1:
                return False
        d += 1 if d == 2 else 2
    if temp > 1:
        return False  # remaining prime factor with exponent 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute powerful numbers up to a moderate limit
    L = 2 * 10**8
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    powerfuls_set = powerfuls  # use the set for O(1) lookup
    
    elapsed_gen = time.time() - start_time
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to numpy for vectorized pair counting within the precomputed range
    P = np.array(powerfuls_list, dtype=np.int64)
    max_sum = 2 * L
    
    # Use chunked bincount to count sums
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    chunk_size = 2000
    
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        chunk = P[i_start:i_end, None] + P[None, :]
        flat = chunk.ravel()
        chunk_counts = np.bincount(flat, minlength=max_sum + 1)
        counts += chunk_counts[:max_sum + 1]
        
        if time.time() > deadline - 40:
            break
    
    # Find top candidates from the precomputed range
    k = min(50, len(counts))
    top_indices = np.argpartition(counts, -k)[-k:]
    
    best_N = 2
    best_pairs = [[1, 1]]
    
    for N in top_indices:
        N = int(N)
        if N < 2:
            continue
        pairs = []
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
    
    # Now try to find better N in higher ranges using targeted search
    # Focus on N values that are sums of powerful numbers where at least one
    # is "large" (beyond our precomputed limit)
    
    # Strategy: For each powerful number a in our precomputed set, 
    # and for various multipliers, check if a * k + a' is powerful
    # This is complex, so instead we'll do a direct search for N in 
    # specific ranges by checking if N - p is powerful for p in powerfuls
    
    # Try candidate N values that are 2 * powerful (since 1 is powerful,
    # N = 2*x where x is powerful gives pair (x, x))
    # But we want distinct pairs, so focus on N where multiple decompositions exist
    
    # Alternative: Use the structure. If N = a + b where a, b powerful,
    # and a = u^2 * v^3, b = w^2 * z^3, then N = u^2 v^3 + w^2 z^3.
    # For small v, z (say v, z <= 100), we can enumerate u, w.
    
    # Let's try a different approach: for small powerful numbers p (from our set),
    # and for each candidate multiplier, check N = p * m + q where q is also powerful.
    # This is still complex.
    
    # Simpler: just try to extend our search by checking if for top N candidates,
    # there are additional pairs where one element > L.
    
    # For the best N found so far, check if there are powerful numbers > L 
    # that sum with a smaller powerful number to N.
    # If N <= 2*L, then any powerful a <= N must be <= 2*L. If a > L, 
    # then a = s^2 * t^3 where t^3 > L/4 (roughly), so t > (L/4)^(1/3).
    # For L = 2*10^8, t > ~37. So we can enumerate t from 37 to (N)^(1/3)
    # and for each t, enumerate s such that s^2 * t^3 <= N.
    
    # Let's implement a function to check if n is powerful for large n
    # and use it to extend our search for the top candidates.
    
    # Re-check top candidates with extended powerful check
    extended_best_N = best_N
    extended_best_pairs = best_pairs
    
    for N in top_indices[:20]:  # check top 20 candidates
        N = int(N)
        if N < 2 or N > 2 * L:
            continue
        
        all_pairs = []
        
        # First, pairs where both a, b <= L
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                all_pairs.append([a, b])
        
        # Now, pairs where a > L (and thus b < N - L)
        # a = s^2 * t^3, with a > L
        # t^3 > L / s^2, so for small s, t must be large
        # Enumerate t from 1 to N^(1/3)
        max_t = int(N ** (1.0/3.0)) + 1
        found_large = set()
        
        for t in range(1, max_t + 1):
            t3 = t * t * t
            # a = s^2 * t^3 > L => s^2 > L / t^3
            min_s = isqrt(L // t3) + 1 if t3 <= L else 1
            max_s = isqrt(N // t3)
            
            for s in range(min_s, max_s + 1):
                a = s * s * t3
                if a > N:
                    break
                b = N - a
                if b < 1:
                    continue
                # Check if b is powerful
                if b in powerfuls_set:
                    if a <= b:
                        pair = [a, b]
                    else:
                        pair = [b, a]
                    if pair not in all_pairs:
                        all_pairs.append(pair)
                        found_large.add(a)
        
        if len(all_pairs) > len(extended_best_pairs):
            extended_best_pairs = all_pairs
            extended_best_N = N
    
    if not extended_best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    extended_best_pairs.sort()
    
    return {'N': extended_best_N, 'pairs': extended_best_pairs}
