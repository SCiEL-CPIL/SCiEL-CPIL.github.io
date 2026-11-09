import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _subset_sum_sets(values):
    """Compute subset sums indexed by cardinality.
    Returns dict: cardinality -> set of achievable sums.
    """
    # dp[k] is a set of achievable sums using k elements
    n = len(values)
    max_sum = sum(values)
    # For efficiency, we can use bitsets, but for small n sets are fine
    # Actually, let's use a list of sets for cardinality 0..n
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for v in values:
        # iterate backwards to avoid using v multiple times in same step
        for k in range(n, 0, -1):
            new_sums = set()
            for s in dp[k - 1]:
                new_sums.add(s + v)
            dp[k] = dp[k] | new_sums
    return dp


def _is_valid(values):
    """Check if two subsets with different cardinalities have different sums.
    i.e., for all i < j, sums_cardinality_i ∩ sums_cardinality_j = empty.
    """
    n = len(values)
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for v in values:
        for k in range(n, 0, -1):
            for s in dp[k - 1]:
                dp[k].add(s + v)
    # Check pairwise disjointness for different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if dp[i] & dp[j]:
                return False
    return True


def _try_extend(current, remaining_pool, target_size, time_limit):
    """Try to extend current set by adding elements from remaining_pool
    to reach target_size while maintaining validity.
    Returns the extended list or None if failed.
    """
    start_time = time.time()
    
    def backtrack(current, remaining, depth):
        if time.time() - start_time > time_limit:
            return None
        if len(current) == target_size:
            return list(current)
        if not remaining:
            return None
        if len(current) + len(remaining) < target_size:
            return None
        
        # Try each candidate in random order
        candidates = list(remaining)
        random.shuffle(candidates)
        
        for idx, v in enumerate(candidates):
            new_current = current + [v]
            new_remaining = remaining[:idx] + remaining[idx+1:]
            
            # Quick check: does adding v violate the condition?
            # We can do a fast check by computing subset sums incrementally,
            # but for simplicity in this backtracking, we call _is_valid
            # which is expensive. Let's optimize by maintaining dp.
            # For now, let's just check.
            if _is_valid(new_current):
                result = backtrack(new_current, new_remaining, depth + 1)
                if result is not None:
                    return result
        return None
    
    return backtrack(list(current), list(remaining_pool), 0)


def _greedy_construct(pool, target_size, time_limit):
    """Greedy construction: try to add elements one by one."""
    current = []
    remaining = list(pool)
    random.shuffle(remaining)
    start_time = time.time()
    
    while len(current) < target_size and remaining:
        if time.time() - start_time > time_limit:
            break
        # Try to add each remaining element
        best_idx = -1
        best_v = None
        for idx, v in enumerate(remaining):
            new_current = current + [v]
            if _is_valid(new_current):
                best_idx = idx
                best_v = v
                break
        
        if best_v is not None:
            current.append(best_v)
            remaining.pop(best_idx)
        else:
            break
    
    return current


def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin
    
    # Pool of candidates: 1..256
    pool = list(range(1, 257))
    
    # We want to maximize size. The theoretical max is 60.
    # But with values up to 256, can we get large sets?
    # The condition is strict: sums of different cardinalities must be disjoint.
    # This is equivalent to: the generating function P(x) = prod(1 + x^a_i) has
    # no coefficient appearing in two different powers of y (if we track cardinality).
    # Or simpler: the sets of subset sums S_k are pairwise disjoint for k != l.
    
    # Let's start with a reasonable target and try to achieve it.
    # We can try to find a set of size 10, 11, etc.
    
    best_set = []
    
    # First, try a greedy approach to get a baseline
    # Then, try to improve with randomized restarts
    
    for target_size in range(9, 13):  # Try sizes 9 to 12
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random restarts
        found = None
        for attempt in range(50):
            if time.time() - start_time > deadline:
                break
            
            random.shuffle(pool)
            # Greedy first
            candidate = _greedy_construct(pool, target_size, time_limit=1.0)
            if len(candidate) >= target_size:
                found = candidate
                break
        
        if found:
            if len(found) > len(best_set):
                best_set = found
            # Try to extend further?
            # For now, just keep the best found
        else:
            # If greedy didn't find it, maybe try a different strategy?
            # Or just accept that this size is hard.
            pass
    
    # If we found a set, return it. Otherwise, return a small valid set.
    if not best_set:
        # Fallback: try to find any valid set
        # e.g., powers of 2 are valid?
        # {1, 2, 4, 8, ...} -> subset sums are unique, so definitely disjoint by cardinality?
        # Wait, unique subset sums implies disjoint by cardinality.
        # Powers of 2 up to 256: 1,2,4,8,16,32,64,128 (8 elements)
        powers = [1, 2, 4, 8, 16, 32, 64, 128]
        if _is_valid(powers):
            best_set = powers
    
    # Ensure we return a list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
