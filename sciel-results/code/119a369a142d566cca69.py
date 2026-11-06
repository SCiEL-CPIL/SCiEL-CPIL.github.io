import time
from math import lcm
import bisect

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i

def _subset_sum_exists(target, elements):
    """
    Check if target can be formed by a non-empty subset of elements.
    elements is a list of integers (scaled reciprocals).
    target is an integer (scaled reciprocal).
    Uses Meet-in-the-Middle.
    """
    if not elements:
        return False
    if target < 0:
        return False
    
    n = len(elements)
    if n == 0:
        return False
    
    # If n is small, simple set DP is fine and faster overhead
    if n <= 10:
        reachable = {0}
        for e in elements:
            new_reachable = set()
            for s in reachable:
                if s <= target:
                    new_reachable.add(s)
                if s + e <= target:
                    new_reachable.add(s + e)
            reachable = new_reachable
        return target in reachable and target != 0 # target != 0 is implied by problem, but safe
    
    # Meet in the middle
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]
    
    # Generate subset sums for left
    sums_left = [0]
    for e in left:
        new_sums = [s + e for s in sums_left if s + e <= target]
        sums_left.extend(new_sums)
    
    # Generate subset sums for right
    sums_right = [0]
    for e in right:
        new_sums = [s + e for s in sums_right if s + e <= target]
        sums_right.extend(new_sums)
        
    # Sort right sums for binary search
    sums_right.sort()
    
    # Check for match
    # We need s_l + s_r == target
    # s_r == target - s_l
    # We must ensure the subset is not empty.
    # If s_l == 0 and s_r == 0, subset is empty.
    # But target > 0, so s_l + s_r = target > 0 implies at least one is non-zero, 
    # so the combined subset is non-empty.
    for s_l in sums_left:
        if s_l > target:
            continue
        needed = target - s_l
        # Binary search needed in sums_right
        idx = bisect.bisect_left(sums_right, needed)
        if idx < len(sums_right) and sums_right[idx] == needed:
            return True
            
    return False

def _is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    S is a list of integers.
    """
    for k in S:
        target = _R[k]
        others = [_R[x] for x in S if x != k]
        if _subset_sum_exists(target, others):
            return False
    return True

def construct():
    # Greedy to find a good initial solution
    # Try adding from 24 down to 1
    S_greedy = []
    for k in range(24, 0, -1):
        # Check if adding k is valid
        if _is_valid(S_greedy + [k]):
            S_greedy.append(k)
            
    max_size = len(S_greedy)
    best_set = S_greedy[:]
    
    # Backtracking to find a larger set
    # We iterate through 1..24.
    # State: idx (current element index 1..24), current_set (list of included elements)
    # We want to find a set of size > max_size
    
    def backtrack(idx, current_set, current_size):
        nonlocal max_size, best_set
        
        if _time_remaining() < 2:
            return False
            
        # Pruning: if we can't beat max_size, stop
        if current_size + (24 - idx + 1) <= max_size:
            return False
            
        if idx == 25:
            if current_size > max_size:
                max_size = current_size
                best_set = current_set[:]
            return current_size > max_size
            
        # Option 1: Exclude idx
        if backtrack(idx + 1, current_set, current_size):
            return True
            
        # Option 2: Include idx
        k = idx
        new_set = current_set + [k]
        
        # Check validity of new_set
        # 1. Is 1/k a subsum of current_set?
        target_k = _R[k]
        elems_k = [_R[x] for x in current_set]
        if _subset_sum_exists(target_k, elems_k):
            return backtrack(idx + 1, current_set, current_size)
            
        # 2. Does adding k break validity of any y in current_set?
        # For each y in current_set, check if 1/y is a subsum of (current_set \ {y}) U {k}
        # This is equivalent to:
        #   1/y = 1/k + sum(subset of current_set \ {y, k})
        #   sum(subset of current_set \ {y, k}) = 1/y - 1/k
        # Let diff = _R[y] - _R[k]
        # If diff <= 0, impossible (since sum of positive reciprocals > 0)
        # Else, check if diff is a subsum of [_R[x] for x in current_set if x != y and x != k]
        
        valid = True
        for y in current_set:
            diff = _R[y] - _R[k]
            if diff <= 0:
                continue
            elems_diff = [_R[x] for x in current_set if x != y and x != k]
            if _subset_sum_exists(diff, elems_diff):
                valid = False
                break
                
        if not valid:
            return backtrack(idx + 1, current_set, current_size)
            
        # If valid, recurse with k included
        return backtrack(idx + 1, new_set, current_size + 1)
        
    # Start backtracking
    # To help find large sets faster, we might want to try including elements first?
    # The current structure tries Exclude first, then Include.
    # This tends to find small sets first.
    # For maximization, it's often better to try Include first to get a large candidate early,
    # which updates max_size and prunes more.
    # However, the pruning `current_size + (24 - idx + 1) <= max_size` works regardless.
    # Let's swap to Include first to see if it finds the optimum faster.
    
    def backtrack_incl_first(idx, current_set, current_size):
        nonlocal max_size, best_set
        
        if _time_remaining() < 2:
            return False
            
        if current_size + (24 - idx + 1) <= max_size:
            return False
            
        if idx == 25:
            if current_size > max_size:
                max_size = current_size
                best_set = current_set[:]
            return current_size > max_size
            
        # Option 1: Include idx
        k = idx
        new_set = current_set + [k]
        
        # Check validity
        target_k = _R[k]
        elems_k = [_R[x] for x in current_set]
        can_include = True
        if _subset_sum_exists(target_k, elems_k):
            can_include = False
        else:
            for y in current_set:
                diff = _R[y] - _R[k]
                if diff <= 0:
                    continue
                elems_diff = [_R[x] for x in current_set if x != y and x != k]
                if _subset_sum_exists(diff, elems_diff):
                    can_include = False
                    break
                    
        if can_include:
            if backtrack_incl_first(idx + 1, new_set, current_size + 1):
                return True
                
        # Option 2: Exclude idx
        return backtrack_incl_first(idx + 1, current_set, current_size)
        
    backtrack_incl_first(1, [], 0)
    
    if not best_set:
        return list(range(13, 25))
        
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
