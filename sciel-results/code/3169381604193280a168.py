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

def _can_add(S, x):
    """
    Fast check if x can be added to S.
    1. x is not mean of any subset of S (size >= 2).
    2. No element in S is mean of any subset of S\{z} U {x} (size >= 2).
       This is equivalent to: for all z in S, z is not mean of any subset of S\{z} U {x}.
       Since S is valid, z is not mean of any subset of S\{z}.
       The only new subsets involving x are {x} U T where T subset of S\{z}.
       So we check: does there exist T subset of S\{z}, |T|>=1, such that 
       k*z = x + sum(T) for some k >= 2?
       => k*z - x = sum(T).
    """
    m = len(S)
    
    # Check 1: x is not mean of S
    if m >= 2:
        max_sum = sum(S)
        dp = [0] * (m + 1)
        dp[0] = 1
        for y in S:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << y)
        
        for k in range(2, m + 1):
            target_sum = k * x
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return False

    # Check 2: Existing elements remain valid
    # For each z in S, check if z is mean of subset of (S\{z}) U {x}
    # This requires checking if k*z - x is a subset sum of S\{z} of size k-1 (>=1)
    # for any k >= 2.
    
    # Optimization: If m is small, just do it. If m is large, this is expensive.
    # We can precompute subset sums for S\{z}? No, that's O(n^2).
    # Let's just iterate.
    
    for z in S:
        others = [y for y in S if y != z]
        m_z = len(others)
        if m_z == 0:
            continue
        
        max_sum_z = sum(others)
        dp_z = [0] * (m_z + 1)
        dp_z[0] = 1
        
        for y in others:
            for k in range(m_z - 1, -1, -1):
                if dp_z[k]:
                    dp_z[k+1] |= (dp_z[k] << y)
        
        # k is total subset size in new set.
        # k >= 2.
        # t_prime_size = k - 1.
        # sum(T') = k*z - x.
        # T' subset of others.
        
        for k in range(2, m_z + 2):
            t_size = k - 1
            if t_size < 1 or t_size > m_z:
                continue
            target = k * z - x
            if target < 0 or target > max_sum_z:
                continue
            if (dp_z[t_size] >> target) & 1:
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
        if _can_add(S, x):
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
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding one element
        random.shuffle(available)
        for x in available[:50]:
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
            if _can_add(S, x):
                S.append(x)
                S_set.add(x)
                available.remove(x)
                improved = True
                break
        
        # 2. Try swapping: remove one, add another
        if not improved and len(S) > 0:
            # Pick a random element to remove
            for _ in range(10): # Try 10 random removals
                if time.time() > deadline:
                    break
                i = random.randint(0, len(S) - 1)
                z = S[i]
                candidate_S = S[:i] + S[i+1:]
                
                # Try to add a better element
                # Prefer elements far from existing ones?
                random.shuffle(available)
                for x in available[:20]:
                    if time.time() > deadline:
                        break
                    if _can_add(candidate_S, x):
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
    deadline = start_time + 85.0
    
    best_S = []
    
    # Strategy 1: Multiple greedy constructions with different orderings
    
    # Ordering 1: Random
    for attempt in range(50):
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
            
    # Ordering 4: Powers of 2 first
    if time.time() < deadline:
        powers = [2**i for i in range(0, 10)]
        rest = [x for x in range(1, 513) if x not in set(powers)]
        random.shuffle(rest)
        order = powers + rest
        S = _greedy_construct(order, deadline)
        if len(S) > len(best_S):
            best_S = S

    # Local search on the best set found
    if time.time() < deadline and len(best_S) > 0:
        best_S = _local_search(best_S, deadline)
    
    # Final verification
    if best_S and not _fast_validity_check(best_S):
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    if not best_S:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    return best_S
