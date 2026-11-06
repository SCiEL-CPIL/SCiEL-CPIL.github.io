import time
from math import gcd
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

def _subset_sum_check(values, target):
    """
    Check if target can be formed by sum of a non-empty subset of values.
    values is a list of positive integers.
    Uses meet-in-the-middle for efficiency.
    """
    if not values:
        return False
    n = len(values)
    if target <= 0:
        return False
    
    total = sum(values)
    if target > total:
        return False
    
    # Check single elements
    for v in values:
        if v == target:
            return True
    
    if n <= 1:
        return False
    
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
    
    # Check if any right subset sum + left subset sum = target
    for mask in range(1 << len(right)):
        s_right = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s_right += right[i]
        if target - s_right in left_sums:
            # Ensure non-empty: if s_right == 0 and target == 0, skip (but target > 0 here)
            # If both sides empty, sum is 0, but target > 0 so this won't happen
            return True
    return False

def _is_valid_set(s):
    """
    Check if set s is reciprocal-subsum-free.
    For each pair (a, b) in s with a < b:
      1/b - 1/a should NOT be a subset sum of reciprocals of s \ {a, b}
    Also for each a in s:
      1/a should NOT be a subset sum of reciprocals of s \ {a}
    """
    s_list = sorted(s)
    n = len(s_list)
    
    # Precompute LCM of entire set for efficiency
    # For each element a, check if 1/a is subset sum of others
    for i, a in enumerate(s_list):
        others = [x for x in s_list if x != a]
        if not others:
            continue
        L = _lcm_list(others)
        # 1/a = sum(1/x) => L/a = sum(L/x)
        if L % a != 0:
            continue  # Can't form, so it's fine
        target_int = L // a
        values = [L // x for x in others]
        if _subset_sum_check(values, target_int):
            return False
    
    # For each pair (a, b) with a < b, check if 1/a - 1/b is subset sum of s\{a,b}
    for i in range(n):
        for j in range(i + 1, n):
            a = s_list[i]
            b = s_list[j]
            # 1/a - 1/b = (b - a) / (a * b)
            remaining = [x for x in s_list if x != a and x != b]
            if not remaining:
                continue
            L = _lcm_list(remaining)
            target_num = b - a
            target_den = a * b
            # Need: L * target_num / target_den to be integer
            num = L * target_num
            if num % target_den != 0:
                continue
            target_int = num // target_den
            if target_int <= 0:
                continue
            values = [L // x for x in remaining]
            if _subset_sum_check(values, target_int):
                return False
    
    return True

def _can_add(k, selected):
    """
    Check if we can add k to the selected set while maintaining validity.
    selected is a list of ints.
    """
    # Check 1: Is 1/k a subset sum of reciprocals of selected?
    if selected:
        L = _lcm_list(selected)
        if L % k == 0:
            target_int = L // k
            values = [L // x for x in selected]
            if _subset_sum_check(values, target_int):
                return False
    
    # Check 2: For each x in selected, is (1/x - 1/k) a subset sum of reciprocals of selected \ {x}?
    for x in selected:
        if x >= k:
            # 1/x <= 1/k, so 1/x - 1/k <= 0, can't be sum of positive reciprocals
            continue
        # target = 1/x - 1/k = (k - x) / (x * k)
        remaining = [y for y in selected if y != x]
        if not remaining:
            continue
        L = _lcm_list(remaining)
        target_num = k - x
        target_den = x * k
        num = L * target_num
        if num % target_den != 0:
            continue
        target_int = num // target_den
        if target_int <= 0:
            continue
        values = [L // y for y in remaining]
        if _subset_sum_check(values, target_int):
            return False
    
    return True

def _greedy_build(ordering):
    """Build a set using greedy addition in the given ordering."""
    result = []
    for k in ordering:
        if _can_add(k, result):
            result.append(k)
    return result

def _local_improve(s):
    """Try to improve set s by swapping elements."""
    s = list(s)
    improved = True
    while improved:
        improved = False
        for i in range(len(s)):
            for j in range(1, 25):
                if j in s:
                    continue
                # Try swapping s[i] with j
                new_s = s[:i] + [j] + s[i+1:]
                if _is_valid_set(new_s):
                    # Also try to add more elements
                    new_s_sorted = sorted(new_s)
                    for k in range(1, 25):
                        if k not in new_s and _can_add(k, new_s):
                            new_s.append(k)
                    if len(new_s) > len(s):
                        s = new_s
                        improved = True
                        break
            if improved:
                break
    return s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = []
    best_size = 0
    
    # Multiple greedy orderings
    orderings = [
        list(range(24, 0, -1)),           # Descending
        list(range(1, 25)),               # Ascending
        list(range(12, 25)) + list(range(12, 0, -1)),  # Middle out
        list(range(15, 25)) + list(range(15, 0, -1)),  # Upper middle out
        list(range(8, 25)) + list(range(8, 0, -1)),    # Lower middle out
        # Some random-ish orderings
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    ]
    
    for ordering in orderings:
        if _time_remaining() < 5.0:
            break
        greedy_set = _greedy_build(ordering)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set[:]
    
    # Local improvement on best set
    if _time_remaining() > 10.0:
        improved = _local_improve(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Try more random restarts
    import random
    random.seed(42)
    for _ in range(50):
        if _time_remaining() < 5.0:
            break
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        greedy_set = _greedy_build(random_order)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_set = greedy_set[:]
    
    # Final local improvement
    if _time_remaining() > 10.0:
        improved = _local_improve(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Ensure non-empty
    if not best_set:
        best_set = [1]
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
    print(f"Valid: {_is_valid_set(result)}")
