import time
import random
from itertools import combinations

def has_subset_sum(elements, k, target):
    """Check if there's a subset of exactly k elements from elements summing to target."""
    n = len(elements)
    if k < 0 or k > n:
        return False
    if k == 0:
        return target == 0
    if k == n:
        return sum(elements) == target
    
    # Early bounds
    sorted_e = sorted(elements)
    min_s = sum(sorted_e[:k])
    max_s = sum(sorted_e[-k:])
    if target < min_s or target > max_s:
        return False
    
    # Use complement if beneficial
    ck = n - k
    if ck < k:
        ct = sum(elements) - target
        return has_subset_sum(elements, ck, ct)
    
    if k == 1:
        return target in elements
    if k == 2:
        seen = set()
        for x in elements:
            if (target - x) in seen:
                return True
            seen.add(x)
        return False
    
    # DP: dp[j][s] = whether we can form sum s using j elements
    # To save memory, use 1D array for each j, but we need all j up to k
    # Actually, we only need up to k, and s up to target
    # But target can be up to 24576, and k up to 48
    # This is 48 * 24576 ~ 1.2M booleans, which is acceptable
    
    # But doing this for each candidate is expensive.
    # Alternative: use a set of achievable sums for each size
    # Let's use a list of sets: sums[j] = set of sums achievable with j elements
    sums = [set() for _ in range(k+1)]
    sums[0].add(0)
    
    for e in elements:
        # Update in reverse to avoid using same element twice
        for j in range(k, 0, -1):
            for s in list(sums[j-1]):
                new_s = s + e
                if new_s <= target:  # Prune
                    sums[j].add(new_s)
    
    return target in sums[k]

def can_add(S, c):
    """Check if c can be added to S."""
    S = list(S)
    n = len(S)
    
    # Condition 1: c is not the mean of any subset of S of size >= 2
    for k in range(2, n+1):
        if has_subset_sum(S, k, k * c):
            return False
    
    # Condition 2: no x in S is the mean of a subset of S U {c} \ {x} of size >= 2 that includes c
    for x in S:
        others = [y for y in S if y != x]
        # We need: for some k >= 2, there is a subset T of others of size k-1 such that sum(T) + c = k*x
        # i.e., sum(T) = k*x - c
        for k in range(2, n+1):
            target = k * x - c
            if target < 0:
                continue
            if has_subset_sum(others, k-1, target):
                return False
    
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin
    
    best = []
    best_sz = 0
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        cands = list(range(1, 513))
        random.shuffle(cands)
        
        cur = []
        for c in cands:
            if time.time() > deadline - 3:
                break
            if can_add(cur, c):
                cur.append(c)
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break
    
    return best[:48]
