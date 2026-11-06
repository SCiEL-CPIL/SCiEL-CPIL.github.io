import time
from fractions import Fraction
from math import gcd
from functools import lru_cache
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(nums):
    result = 1
    for n in nums:
        result = _lcm(result, n)
    return result

def _is_subset_sum_integer(target_num, target_den, elements):
    """
    Check if target_num/target_den can be formed by sum of 1/x for x in some non-empty subset of elements.
    Uses integer arithmetic by scaling by LCM of elements.
    """
    if not elements:
        return False
    
    # Scale target to common denominator
    # target = target_num / target_den
    # We need: sum(1/x_i) = target_num/target_den
    # Multiply both sides by LCM of all elements and target_den
    # Let L = LCM(elements)
    # Then sum(L/x_i) = L * target_num / target_den
    # So we need L * target_num to be divisible by target_den, and sum(L/x_i) = L*target_num/target_den
    
    L = _lcm_list(elements)
    
    # Check divisibility
    numerator = L * target_num
    if numerator % target_den != 0:
        return False
    target_int = numerator // target_den
    
    # Now check if target_int is a subset sum of {L//x for x in elements}
    # This is a standard subset sum problem with integers
    
    # Quick checks
    values = [L // x for x in elements]
    total_sum = sum(values)
    if target_int > total_sum:
        return False
    if target_int < min(values):
        return False
    
    # Check if any single element matches
    for v in values:
        if v == target_int:
            return True
    
    # Use meet-in-the-middle for subset sum if n is reasonable
    n = len(values)
    if n <= 20:
        # Meet in the middle
        half = n // 2
        left = values[:half]
        right = values[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # Check if any right subset sum + left subset sum = target_int
        for mask in range(1 << len(right)):
            s_right = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s_right += right[i]
            if target_int - s_right in left_sums:
                # Make sure we're not using empty set from both sides
                # But since we checked single elements above, and target_int > 0,
                # at least one side must be non-empty if target_int > 0
                # Actually we need to ensure the combined subset is non-empty
                # If s_right == 0 and target_int - s_right == 0, that's empty
                if s_right == 0 and target_int == 0:
                    continue
                return True
        return False
    else:
        # For larger n, use a simpler DFS with pruning
        values_sorted = sorted(values, reverse=True)
        
        def dfs(idx, current_sum):
            if current_sum == target_int:
                return True
            if idx >= n:
                return False
            if current_sum > target_int:
                return False
            # Pruning: even if we take all remaining, can't reach target
            remaining = sum(values_sorted[idx:])
            if current_sum + remaining < target_int:
                return False
            
            # Include
            if dfs(idx + 1, current_sum + values_sorted[idx]):
                return True
            # Exclude
            if dfs(idx + 1, current_sum):
                return True
            return False
        
        return dfs(0, 0)

def _can_add(k, selected):
    """
    Check if we can add k to the selected set.
    selected is a list of ints.
    """
    # Check 1: Is 1/k a subset sum of reciprocals of selected?
    # target = 1/k, so target_num=1, target_den=k
    if _is_subset_sum_integer(1, k, selected):
        return False
        
    # Check 2: For each x in selected, is (1/x - 1/k) a subset sum of reciprocals of selected \ {x}?
    for x in selected:
        # If 1/x <= 1/k, then 1/x - 1/k <= 0, so it cannot be a sum of positive reciprocals
        if x >= k:
            continue
            
        # target = 1/x - 1/k = (k - x) / (x * k)
        target_num = k - x
        target_den = x * k
        remaining = [y for y in selected if y != x]
        if _is_subset_sum_integer(target_num, target_den, remaining):
            return False
            
    return True

def _validate_set(s):
    """Validate that set s is reciprocal-subsum-free."""
    s_list = list(s)
    for i, k in enumerate(s_list):
        # Check 1: 1/k not in subset sums of others
        others = [x for x in s_list if x != k]
        if _is_subset_sum_integer(1, k, others):
            return False
        # Check 2: For each x in others, 1/x - 1/k not in subset sums of others \ {x}
        for x in others:
            if x >= k:
                continue
            target_num = k - x
            target_den = x * k
            remaining = [y for y in others if y != x]
            if _is_subset_sum_integer(target_num, target_den, remaining):
                return False
    return True

def _try_additions(current_set, candidates, best_size):
    """Try to add candidates to current_set, return improved set if possible."""
    improved = list(current_set)
    for k in candidates:
        if k not in improved and _can_add(k, improved):
            improved.append(k)
            if len(improved) > best_size:
                return improved
    return improved

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = []
    best_size = 0
    
    # Greedy initialization with multiple orderings
    orderings = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        list(range(12, 25)) + list(range(12, 0, -1)),  # Middle out
    ]
    
    for ordering in orderings:
        greedy_set = []
        for k in ordering:
            if _can_add(k, greedy_set):
                greedy_set.append(k)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set[:]
    
    # Branch and bound search
    elements = list(range(24, 0, -1))
    
    def dfs(idx, current_set):
        nonlocal best_set, best_size
        
        # Check deadline
        if _time_remaining() < 1.0:
            return
            
        # Pruning: If we can't beat the best size, return
        if len(current_set) + (24 - idx) <= best_size:
            return
            
        if idx == 24:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        k = elements[idx]
        
        # Try including elements[idx]
        if _can_add(k, current_set):
            current_set.append(k)
            dfs(idx + 1, current_set)
            current_set.pop()
            
        # Try excluding elements[idx]
        dfs(idx + 1, current_set)

    dfs(0, [])
    
    # If time remains, try local improvement on best_set
    if _time_remaining() > 5.0:
        candidates = [x for x in range(1, 25) if x not in best_set]
        improved = _try_additions(best_set, candidates, best_size)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    return best_set

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
