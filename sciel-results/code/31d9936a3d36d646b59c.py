"""Rewritten program for the suite task; deterministic greedy construction with fast validation."""
import itertools
import random

# EVOLVE-BLOCK-START

def _check_validity(sums_set):
    """
    Check if a set of subset sums satisfies the no-divisibility condition.
    sums_set: set of integers.
    Returns True if valid, False otherwise.
    """
    # Optimization: if 1 is in sums, it divides everything. So 1 cannot be in the set unless it's the only sum?
    # But 1 is a subset sum if 1 is in the original set.
    # If 1 is in the original set, sum=1 exists. 1 divides any other sum.
    # So if len(sums_set) > 1 and 1 in sums_set, return False.
    if 1 in sums_set and len(sums_set) > 1:
        return False
        
    # Check divisibility
    # To speed up, we can sort and only check divisors.
    # But for small sets (n<=15, sums <= 2^15 ~ 32k), O(S^2) might be too slow in pure Python if S is large.
    # However, S is the number of subset sums, which is 2^n - 1.
    # If n=15, S=32767. S^2 is 1e9. Too slow.
    
    # We need a faster check.
    # If s1 divides s2, then s2 = k * s1.
    # We can iterate through s1 in sums_set, and check if any multiple of s1 is in sums_set.
    # Max sum is sum(nums) <= 15 * 256 = 3840.
    # So for each s1, we check k*s1 for k=2..max_sum/s1.
    
    max_sum = max(sums_set)
    for s1 in sums_set:
        if s1 == 1:
            # 1 divides everything. If there's any other sum, invalid.
            if len(sums_set) > 1:
                return False
            continue
        # Check multiples
        multiple = 2 * s1
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s1
    return True

def _try_add_element(current_nums, current_sums, candidate):
    """
    Try to add candidate to current_nums.
    Returns (new_nums, new_sums) if valid, else None.
    """
    new_sums = set(current_sums)
    new_sums.add(candidate)
    
    # Check for duplicate sums immediately
    # New sums are candidate and s + candidate for s in current_sums
    # If candidate is already in current_sums, duplicate.
    if candidate in current_sums:
        return None
        
    for s in current_sums:
        new_sum = s + candidate
        if new_sum in new_sums:
            return None
        new_sums.add(new_sum)
        
    # Now check divisibility for the new set of sums
    if _check_validity(new_sums):
        return current_nums + [candidate], new_sums
    else:
        return None

def construct():
    """
    Return a list of distinct integers in 1..256 satisfying the condition, maximizing size.
    Uses a greedy approach with multiple random restarts to find a large valid set.
    """
    best_set = []
    best_size = 0
    
    # Try several random orders of candidates
    candidates = list(range(1, 257))
    
    # We'll do a few passes with different random seeds/orders
    for seed in range(10):
        random.seed(seed)
        random.shuffle(candidates)
        
        current_nums = []
        current_sums = set()
        
        for c in candidates:
            res = _try_add_element(current_nums, current_sums, c)
            if res:
                current_nums, current_sums = res
                if len(current_nums) > best_size:
                    best_size = len(current_nums)
                    best_set = current_nums[:]
                    
        # Also try descending order in a separate pass or as part of the shuffle?
        # The shuffle covers it.
        
        # Early stop if we reach a very high size, e.g., 15
        if best_size >= 15:
            break

    # Ensure we return a list of Python ints
    result = [int(x) for x in best_set]
    
    # Final verification
    if result:
        # Recompute sums to be absolutely sure
        n = len(result)
        sums = set()
        for i in range(1, 1 << n):
            s = 0
            for j in range(n):
                if i & (1 << j):
                    s += result[j]
            if s in sums:
                # Invalid, fallback to empty or known small set
                # But our greedy should prevent this.
                # If this happens, we return a known valid small set
                return [31, 32, 34]
            sums.add(s)
        
        if not _check_validity(sums):
            return [31, 32, 34]
            
    return result

# EVOLVE-BLOCK-END
