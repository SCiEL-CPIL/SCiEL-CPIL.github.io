"""
Strategy: Target-Specific Enumeration for powerful number pair counting.
1. Use sparse pairwise summation with np.bincount on a moderate L to find top candidate N values.
2. For each top candidate N, verify and extend the pair count by checking powerful numbers 
   in a wider range [1, N-1] using direct powerful-number tests.
3. Return the candidate with the most verified pairs.
This avoids O(P^2) memory for very large L while still exploring higher N ranges.
"""

# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
import numpy as np
import bisect
from collections import defaultdict

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    
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
    """Check if n is a powerful number by exact factorization."""
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count == 1:
                return False
        d += 1 if d == 2 else 2  # Skip even numbers after 2
    if temp > 1:
        return False  # temp is a prime factor appearing once
    return True

def _count_pairs_for_N(N, powerfuls_set, powerfuls_list, max_check=None):
    """Count pairs (a, b) with a <= b, a + b = N, both powerful.
    If max_check is given, only check a up to max_check (and b = N - a)."""
    count = 0
    pairs = []
    half_N = N // 2
    if max_check is not None:
        upper = min(half_N, max_check)
    else:
        upper = half_N
    
    # Use bisect to find the range of powerfuls_list to check
    end_idx = bisect.bisect_right(powerfuls_list, upper)
    for i in range(end_idx):
        a = powerfuls_list[i]
        b = N - a
        if b < a:
            break
        if b in powerfuls_set:
            count += 1
            pairs.append([int(a), int(b)])
    return count, pairs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Phase 1: Find top candidate N values using sparse search on moderate L
    # Start with L = 10^8 for good coverage
    L_search = 100_000_000
    
    powerfuls = _generate_powerfuls(L_search)
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls_list)
    pows_arr = np.array(powerfuls_list, dtype=np.int32)
    P = len(pows_arr)
    
    elapsed_gen = time.time() - start_time
    
    if P > 50000 or time.time() > deadline - 60:
        # Reduce L if too many powerfuls or not enough time
        L_search = 30_000_000
        powerfuls = _generate_powerfuls(L_search)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        pows_arr = np.array(powerfuls_list, dtype=np.int32)
        P = len(pows_arr)
    
    if time.time() > deadline - 50:
        # Fallback: return best from small brute force
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
    
    # Compute all pair sums and count frequencies
    try:
        i_idx, j_idx = np.triu_indices(P)
        sums = pows_arr[i_idx] + pows_arr[j_idx]
        max_sum = 2 * L_search
        counts = np.bincount(sums, minlength=max_sum + 1)
        
        # Get top K candidate N values by count
        K = 20  # Top 20 candidates
        # Find indices of top K counts
        top_indices = np.argsort(counts)[::-1][:K]
        top_indices = sorted([int(x) for x in top_indices if x >= 2])
        
    except MemoryError:
        # Fallback to smaller L
        L_search = 10_000_000
        powerfuls = _generate_powerfuls(L_search)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        pows_arr = np.array(powerfuls_list, dtype=np.int32)
        P = len(pows_arr)
        
        i_idx, j_idx = np.triu_indices(P)
        sums = pows_arr[i_idx] + pows_arr[j_idx]
        max_sum = 2 * L_search
        counts = np.bincount(sums, minlength=max_sum + 1)
        top_indices = np.argsort(counts)[::-1][:20]
        top_indices = sorted([int(x) for x in top_indices if x >= 2])
    
    elapsed_phase1 = time.time() - start_time
    
    # Phase 2: For each top candidate, try to extend the search to larger N
    # by checking if larger powerful numbers can also form pairs summing to N.
    # Strategy: For candidate N, we know pairs with a, b <= L_search.
    # But there might be pairs where a > L_search or b > L_search.
    # Since a + b = N, if a > L_search then b = N - a < N - L_search.
    # So we need to check powerful numbers in (L_search, N) for a, and 
    # powerful numbers in (0, N - L_search) for b.
    
    # However, generating all powerfuls up to N could be expensive if N is large.
    # Instead, we'll use a hybrid: check a in [1, min(L_search, N//2)] from our list,
    # and for a > L_search, we need b = N - a < N - L_search.
    # If N - L_search <= L_search, then b <= L_search, so b is in our set.
    # We just need to check if a = N - b is powerful for b in [N - L_search, N//2].
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    for N in top_indices:
        if time.time() > deadline - 20:
            break
        
        # First, count pairs where both a and b <= L_search (already in our set)
        count_small, pairs_small = _count_pairs_for_N(N, powerfuls_set, powerfuls_list, max_check=L_search)
        
        # Now, check for pairs where a > L_search (so b < N - L_search)
        # b must be in (0, N - L_search) and powerful
        # a = N - b must be powerful and > L_search
        extra_count = 0
        extra_pairs = []
        
        if N > 2 * L_search:
            # Then a > L_search implies b < N - L_search, but b could be up to N - L_search
            # Actually if N > 2*L_search, then a > L_search and b = N - a < N - L_search
            # b ranges from 1 to N - L_search - 1 (approximately)
            # We need b to be powerful and a = N - b to be powerful
            # b < N - L_search, so b is in (0, N - L_search)
            # We can iterate over b in powerfuls_list that are < N - L_search
            # and check if N - b is powerful and > L_search
            
            max_b = N - L_search - 1
            if max_b > 0:
                end_idx = bisect.bisect_right(powerfuls_list, max_b)
                for i in range(end_idx):
                    b = powerfuls_list[i]
                    a = N - b
                    if a <= L_search:
                        continue  # Already counted in small pairs
                    if a > N:
                        break
                    if _is_powerful(a):
                        extra_count += 1
                        extra_pairs.append([int(b), int(a)] if b < a else [int(a), int(b)])
        else:
            # N <= 2 * L_search, so a > L_search implies b < N - L_search <= L_search
            # b is in (0, N - L_search) and in our powerfuls_list
            # a = N - b > L_search, need to check if a is powerful
            max_b = N - L_search - 1
            if max_b > 0:
                end_idx = bisect.bisect_right(powerfuls_list, max_b)
                for i in range(end_idx):
                    b = powerfuls_list[i]
                    a = N - b
                    if a <= L_search:
                        continue
                    if _is_powerful(a):
                        extra_count += 1
                        extra_pairs.append([int(b), int(a)] if b < a else [int(a), int(b)])
        
        total_count = count_small + extra_count
        total_pairs = pairs_small + extra_pairs
        
        if total_count > best_count:
            best_count = total_count
            best_N = N
            best_pairs = total_pairs
    
    # Also, consider that the true best N might be larger than 2 * L_search
    # and wasn't in our top candidates. Let's try some large N values.
    # Heuristic: N values that are sums of many small powerfuls tend to be
    # "round" numbers. Try N = 2 * L_search + delta for various delta.
    
    if time.time() < deadline - 30:
        # Try a few large N values
        # The maximum N we can check efficiently is limited by the cost of _is_powerful
        # Let's try N values around 2 * L_search
        for delta in range(0, 100000, 1000):
            N_large = 2 * L_search + delta
            if N_large > 10**18:
                break
            
            # Count pairs: a + b = N_large, a <= b
            # a ranges from 1 to N_large // 2
            # If a <= L_search, b = N_large - a >= L_search
            # If a > L_search, b < L_search
            
            count_large = 0
            pairs_large = []
            
            # Part 1: a in [1, min(L_search, N_large // 2)]
            max_a1 = min(L_search, N_large // 2)
            end_idx1 = bisect.bisect_right(powerfuls_list, max_a1)
            for i in range(end_idx1):
                a = powerfuls_list[i]
                b = N_large - a
                if b < a:
                    break
                if b <= L_search:
                    if b in powerfuls_set:
                        count_large += 1
                        pairs_large.append([int(a), int(b)])
                else:
                    if _is_powerful(b):
                        count_large += 1
                        pairs_large.append([int(a), int(b)])
            
            # Part 2: a in (L_search, N_large // 2], so b in [1, L_search)
            if N_large // 2 > L_search:
                # b in [1, N_large - L_search - 1] but b <= L_search
                # Actually b = N_large - a, a > L_search => b < N_large - L_search
                # b >= 1, b <= L_search
                max_b2 = min(L_search, N_large - L_search - 1)
                end_idx2 = bisect.bisect_right(powerfuls_list, max_b2)
                for i in range(end_idx2):
                    b = powerfuls_list[i]
                    a = N_large - b
                    if a <= L_search:
                        continue
                    if a > N_large // 2:
                        break
                    if _is_powerful(a):
                        count_large += 1
                        pairs_large.append([int(b), int(a)] if b < a else [int(a), int(b)])
            
            if count_large > best_count:
                best_count = count_large
                best_N = N_large
                best_pairs = pairs_large
    
    if best_pairs:
        return {'N': best_N, 'pairs': best_pairs}
    else:
        return {'N': 2, 'pairs': [[1, 1]]}

# EVOLVE-BLOCK-END
