"""
Strategy: Sparse pairwise summation with memory-efficient bincount for powerful number pair counting.
This approach generates all powerful numbers up to a limit L, then computes the sum of every pair (i, j) with i <= j using vectorized numpy operations. Instead of storing a dense 2D matrix, we use np.triu_indices to generate the upper-triangular index pairs and compute their sums in a single vectorized operation. We then use np.bincount to efficiently count how many times each sum appears, which gives us the number of representations N = a + b where a and b are powerful numbers with a <= b.

Key optimizations:
1. Use int32 for sums when possible (L < 2^31/2) to halve memory usage
2. Generate powerful numbers efficiently using the a^2 * b^3 parametrization
3. Use np.bincount (O(N) C implementation) instead of np.unique (O(N log N)) for counting
4. Fallback to smaller L if memory or time constraints are hit
5. Extract pairs for the best N using binary search and set membership

For L=10^8, P ~ 15000-20000 powerful numbers, P^2/2 ~ 1-2*10^8 pairs.
With int32: 1-2*10^8 * 4 bytes = 400MB-800MB for sums, plus similar for indices.
This is feasible within typical memory limits and the 150s time budget.
"""

# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np
import bisect

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    # b must be square-free and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    
    # Primes up to sqrt(max_b) for square-free check
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    
    return powerfuls

def _is_powerful(n):
    """Check if n is powerful by exact factorization."""
    if n <= 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt == 1:
                return False
        d += 1 if d == 2 else 2
    if temp > 1:
        return False  # remaining prime factor has exponent 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Strategy: Target-specific enumeration
    # Pick a highly composite base B (product of small prime powers)
    # Generate powerful numbers of form B * k^2, which are automatically powerful
    # since B is powerful and k^2 is a square.
    
    # B = (2*3*5*7*11)^2 = 210^2 = 44100
    # But we can also use B = 2^2 * 3^2 * 5^2 * 7^2 = 44100
    # Let's use multiple bases to get more powerful numbers
    
    L = 100_000_000  # Limit for powerful numbers
    
    # Generate all powerful numbers up to L
    powerfuls = _generate_powerfuls(L)
    
    # Additionally, generate powerful numbers of form B * k^2 for various B
    # where B is a powerful number (product of prime powers with exponent >= 2)
    # This ensures we capture more powerful numbers efficiently
    
    # Try several bases: products of squares of small primes
    bases = []
    # 2^2 = 4
    bases.append(4)
    # 2^2 * 3^2 = 36
    bases.append(36)
    # 2^2 * 3^2 * 5^2 = 900
    bases.append(900)
    # 2^2 * 3^2 * 5^2 * 7^2 = 44100
    bases.append(44100)
    # 2^2 * 3^2 * 5^2 * 7^2 * 11^2 = 5336100
    bases.append(5336100)
    # 2^2 * 3^2 * 5^2 * 7^2 * 11^2 * 13^2 = 902380500 (too large, skip)
    
    for B in bases:
        if B > L:
            continue
        max_k = isqrt(L // B)
        for k in range(1, max_k + 1):
            val = B * k * k
            if val <= L:
                powerfuls.add(val)
    
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls_list)
    pows_arr = np.array(powerfuls_list, dtype=np.int64)
    P = len(pows_arr)
    
    elapsed_gen = time.time() - start_time
    
    if time.time() > deadline - 30:
        # Not enough time left, return a small valid result
        small_limit = 10000
        small_pows = [p for p in powerfuls_list if p <= small_limit]
        count_dict = {}
        pair_dict = {}
        for i in range(len(small_pows)):
            a = small_pows[i]
            for j in range(i, len(small_pows)):
                b = small_pows[j]
                s = a + b
                if s > small_limit * 2:
                    break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use int32 if possible to save memory
    max_sum = 2 * L
    if max_sum <= 2**31 - 1:
        pows_arr_int32 = pows_arr.astype(np.int32)
        use_int32 = True
    else:
        use_int32 = False
    
    try:
        # Generate upper-triangular indices
        i_idx, j_idx = np.triu_indices(P)
        
        # Compute sums
        if use_int32:
            sums = pows_arr_int32[i_idx] + pows_arr_int32[j_idx]
        else:
            sums = pows_arr[i_idx] + pows_arr[j_idx]
        
        # Count frequencies using bincount
        counts = np.bincount(sums, minlength=max_sum + 1)
        
        # Find the sum with maximum count
        best_N = int(np.argmax(counts))
        
        if best_N < 2:
            best_N = 2
        
        # Extract pairs for best_N
        pairs = []
        half_N = best_N // 2
        end_idx = bisect.bisect_right(powerfuls_list, half_N)
        
        for a in powerfuls_list[:end_idx]:
            b = best_N - a
            if b in powerfuls_set:
                if a <= b:
                    pairs.append([int(a), int(b)])
        
        if pairs:
            return {'N': best_N, 'pairs': pairs}
        else:
            return {'N': 2, 'pairs': [[1, 1]]}
            
    except MemoryError:
        # Fall back to smaller L
        L = 30_000_000
        powerfuls = _generate_powerfuls(L)
        
        # Add B*k^2 forms
        for B in bases:
            if B > L:
                continue
            max_k = isqrt(L // B)
            for k in range(1, max_k + 1):
                val = B * k * k
                if val <= L:
                    powerfuls.add(val)
        
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls)
        pows_arr = np.array(powerfuls_list, dtype=np.int64)
        P = len(pows_arr)
        
        if P == 0:
            return {'N': 2, 'pairs': [[1, 1]]}
        
        max_sum = 2 * L
        use_int32 = max_sum <= 2**31 - 1
        
        try:
            i_idx, j_idx = np.triu_indices(P)
            if use_int32:
                pows_arr_int32 = pows_arr.astype(np.int32)
                sums = pows_arr_int32[i_idx] + pows_arr_int32[j_idx]
            else:
                sums = pows_arr[i_idx] + pows_arr[j_idx]
            
            counts = np.bincount(sums, minlength=max_sum + 1)
            best_N = int(np.argmax(counts))
            if best_N < 2:
                best_N = 2
            
            pairs = []
            half_N = best_N // 2
            end_idx = bisect.bisect_right(powerfuls_list, half_N)
            for a in powerfuls_list[:end_idx]:
                b = best_N - a
                if b in powerfuls_set:
                    if a <= b:
                        pairs.append([int(a), int(b)])
            
            if pairs:
                return {'N': best_N, 'pairs': pairs}
        except MemoryError:
            pass
        
        return {'N': 2, 'pairs': [[1, 1]]}
    except Exception:
        return {'N': 2, 'pairs': [[1, 1]]}

# EVOLVE-BLOCK-END
