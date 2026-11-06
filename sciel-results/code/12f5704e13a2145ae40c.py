import time
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is powerful: every prime factor appears with exponent >= 2."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    for p in range(2, isqrt(temp) + 1):
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp < 2:
                return False
    if temp > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    L = 10**8
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
    powerfuls_set = powerfuls
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    P = np.array(powerfuls_list, dtype=np.int64)
    max_sum = 2 * L
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    chunk_size = 1000
    
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        chunk = P[i_start:i_end, None] + P[None, :]
        flat = chunk.ravel()
        chunk_counts = np.bincount(flat, minlength=max_sum + 1)
        counts += chunk_counts[:max_sum + 1]
        
        if time.time() > deadline - 50:
            break
    
    k = min(100, len(counts))
    top_indices = np.argpartition(counts, -k)[-k:]
    top_indices = top_indices[np.argsort(counts[top_indices])[::-1]]
    
    best_N = 2
    best_pairs = [[1, 1]]
    
    for N in top_indices[:60]:
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
        if time.time() > deadline - 30:
            break
    
    extended_best_N = best_N
    extended_best_pairs = best_pairs
    
    for N in top_indices[:20]:
        N = int(N)
        if N < 2 or N > 2 * L:
            continue
        
        all_pairs = []
        seen_pairs = set()
        
        for a in powerfuls_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                all_pairs.append([a, b])
                seen_pairs.add((a, b))
        
        max_t = int(N ** (1.0/3.0)) + 1
        for t in range(1, max_t + 1):
            t3 = t * t * t
            min_s = isqrt(L // t3) + 1 if t3 <= L else 1
            max_s = isqrt(N // t3)
            
            for s in range(min_s, max_s + 1):
                a = s * s * t3
                if a > N:
                    break
                if a <= L:
                    continue
                b = N - a
                if b < 1:
                    continue
                if b in powerfuls_set:
                    if a <= b:
                        pair = (a, b)
                    else:
                        pair = (b, a)
                    if pair not in seen_pairs:
                        all_pairs.append(list(pair))
                        seen_pairs.add(pair)
                elif b > L and b <= N:
                    if _is_powerful(b):
                        if a <= b:
                            pair = (a, b)
                        else:
                            pair = (b, a)
                        if pair not in seen_pairs:
                            all_pairs.append(list(pair))
                            seen_pairs.add(pair)
            
            if time.time() > deadline - 20:
                break
        
        if len(all_pairs) > len(extended_best_pairs):
            extended_best_pairs = all_pairs
            extended_best_N = N
        
        if time.time() > deadline - 20:
            break
    
    if not extended_best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    extended_best_pairs.sort()
    
    return {'N': extended_best_N, 'pairs': extended_best_pairs}
