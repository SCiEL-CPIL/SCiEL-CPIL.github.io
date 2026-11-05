import time
import random
from itertools import combinations

def _fast_validity_check(S):
    """
    Check if set S is mean-free: no element is the arithmetic mean of any
    subset of at least 2 OTHER elements.
    
    Uses dynamic programming with bitsets for subset sums.
    """
    n = len(S)
    if n < 3:
        return True
    
    S_sorted = sorted(S)
    S_set = set(S)
    
    for x in S_sorted:
        others = [y for y in S if y != x]
        m = len(others)
        if m < 2:
            continue
        
        max_sum = sum(others)
        # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for y in others:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << y)
        
        # Check if x is mean of any subset of size k >= 2
        for k in range(2, m + 1):
            target_sum = k * x
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return False
    
    return True

def _incremental_check(S, x):
    """
    Check if adding x to set S (as a list of already-valid elements) keeps it valid.
    We need to check:
    1. x is not the mean of any subset of S of size >= 2
    2. No element in S is the mean of any subset of (S \ {that element}) + {x} of size >= 2
    
    For efficiency, we focus on the most likely violations.
    """
    S_list = list(S)
    m = len(S_list)
    
    # Check 1: x is not mean of any subset of S of size >= 2
    max_sum = sum(S_list)
    dp = [0] * (m + 1)
    dp[0] = 1
    
    for y in S_list:
        for k in range(m - 1, -1, -1):
            if dp[k]:
                dp[k+1] |= (dp[k] << y)
    
    for k in range(2, m + 1):
        target_sum = k * x
        if target_sum > max_sum:
            break
        if (dp[k] >> target_sum) & 1:
            return False
    
    # Check 2: For each element z in S, check if z is mean of subset of (S\{z}) + {x}
    # This is expensive, so we do a lighter check:
    # If z is mean of a subset T of (S\{z}) + {x} of size k >= 2,
    # then either:
    #   a) x is not in T: already checked when S was valid
    #   b) x is in T: then z is mean of {x} U T' where T' subset of S\{z}, |T'| >= 1
    #      So k*z = x + sum(T'), with |T'| >= 1
    #      This means (k*z - x) is achievable as sum of |T'| elements from S\{z}
    
    for z in S_list:
        others_z = [y for y in S_list if y != z]
        m_z = len(others_z)
        if m_z < 1:
            continue
        
        # We need to check if there exists k >= 2 and T' subset of others_z with |T'| >= 1
        # such that k*z = x + sum(T')
        # i.e., sum(T') = k*z - x, with |T'| >= 1
        # and k >= 2
        
        max_sum_z = sum(others_z)
        dp_z = [0] * (m_z + 1)
        dp_z[0] = 1
        
        for y in others_z:
            for k in range(m_z - 1, -1, -1):
                if dp_z[k]:
                    dp_z[k+1] |= (dp_z[k] << y)
        
        for k in range(2, m_z + 2):  # k is the total subset size including x
            t_prime_size = k - 1  # size of T'
            if t_prime_size < 1 or t_prime_size > m_z:
                continue
            target = k * z - x
            if target < 0 or target > max_sum_z:
                continue
            if (dp_z[t_prime_size] >> target) & 1:
                return False
    
    return True

def _greedy_construct(order, deadline):
    """
    Greedily build a valid set by adding elements in the given order.
    Returns the set built.
    """
    S = []
    for x in order:
        if time.time() > deadline:
            break
        if len(S) >= 48:
            break
        if _incremental_check(S, x):
            S.append(x)
    return S

def _local_search(S, deadline):
    """
    Try to improve set S by swapping elements or adding new ones.
    Returns improved set.
    """
    S = list(S)
    S_set = set(S)
    available = [x for x in range(1, 513) if x not in S_set]
    random.shuffle(available)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding one element
        for x in available[:20]:
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
            if _incremental_check(S, x):
                S.append(x)
                available.remove(x)
                S_set.add(x)
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved and len(S) > 0:
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                candidate_S = S[:i] + S[i+1:]
                for x in available[:10]:
                    if time.time() > deadline:
                        break
                    if _incremental_check(candidate_S, x):
                        S = candidate_S + [x]
                        S_set = set(S)
                        available = [y for y in range(1, 513) if y not in S_set]
                        improved = True
                        break
                if improved:
                    break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    best_S = []
    
    # Strategy 1: Multiple greedy constructions with different orderings
    # Use several different ordering strategies
    
    # Ordering 1: Random
    for attempt in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, 513))
        random.shuffle(order)
        S = _greedy_construct(order, deadline)
        if len(S) > len(best_S):
            best_S = S
        if len(best_S) >= 48:
            return best_S
    
    # Ordering 2: Sorted ascending
    if time.time() < deadline:
        S = _greedy_construct(list(range(1, 513)), deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Ordering 3: Sorted descending
    if time.time() < deadline:
        S = _greedy_construct(list(range(512, 0, -1)), deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Ordering 4: Alternating high/low
    if time.time() < deadline:
        low = list(range(1, 257))
        high = list(range(257, 513))
        order = []
        i, j = 0, 0
        while i < len(low) or j < len(high):
            if i < len(low):
                order.append(low[i])
                i += 1
            if j < len(high):
                order.append(high[j])
                j += 1
        S = _greedy_construct(order, deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Ordering 5: Powers of 2 first, then fill
    if time.time() < deadline:
        powers = [2**i for i in range(0, 10)]  # 1, 2, 4, ..., 512
        rest = [x for x in range(1, 513) if x not in set(powers)]
        random.shuffle(rest)
        order = powers + rest
        S = _greedy_construct(order, deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Ordering 6: Fibonacci-like sequence first
    if time.time() < deadline:
        fibs = [1, 2]
        while fibs[-1] + fibs[-2] <= 512:
            fibs.append(fibs[-1] + fibs[-2])
        rest = [x for x in range(1, 513) if x not in set(fibs)]
        random.shuffle(rest)
        order = fibs + rest
        S = _greedy_construct(order, deadline)
        if len(S) > len(best_S):
            best_S = S
    
    # Local search on the best set found
    if time.time() < deadline and len(best_S) > 0:
        best_S = _local_search(best_S, deadline)
    
    # Final verification
    if best_S and not _fast_validity_check(best_S):
        # Fallback to a known valid set
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    if not best_S:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    return best_S
