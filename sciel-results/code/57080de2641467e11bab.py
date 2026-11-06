"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from itertools import combinations

def _lcm(a, b):
    return a * b // _gcd(a, b)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_valid_set(S):
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            if _lcm(S[i], S[j]) <= 256:
                return False
    return True

def _reciprocal_sum(S):
    return sum(1.0 / x for x in S)

def _greedy_initial():
    selected = []
    for x in range(2, 257):
        if all(_lcm(x, y) > 256 for y in selected):
            selected.append(x)
    return selected

def _local_search(initial_set, deadline):
    best_set = list(initial_set)
    best_val = _reciprocal_sum(best_set)
    
    # Precompute conflict matrix
    nums = list(range(2, 257))
    idx = {x: i for i, x in enumerate(nums)}
    n = len(nums)
    # conflict[i][j] = True if lcm(nums[i], nums[j]) <= 256
    conflict = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if _lcm(nums[i], nums[j]) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def _check_add(S, x):
        ix = idx[x]
        for y in S:
            if conflict[ix][idx[y]]:
                return False
        return True
    
    def _check_remove_add(S, to_remove, to_add):
        remaining = [x for x in S if x not in to_remove]
        for x in to_add:
            if not _check_add(remaining, x):
                return False
        # Check among to_add
        for i in range(len(to_add)):
            for j in range(i+1, len(to_add)):
                if conflict[idx[to_add[i]]][idx[to_add[j]]]:
                    return False
        return True
    
    # Local search: try single swaps and small additions/removals
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_set = list(best_set)
        
        # Try adding one number
        for x in nums:
            if x not in best_set and _check_add(best_set, x):
                new_set = best_set + [x]
                new_val = _reciprocal_sum(new_set)
                if new_val > best_val + 1e-12:
                    best_set = new_set
                    best_val = new_val
                    improved = True
        
        # Try removing one number
        for x in best_set[:]:
            new_set = [y for y in best_set if y != x]
            new_val = _reciprocal_sum(new_set)
            # Only remove if it allows adding others later; for now skip pure removals
            # unless it helps in combination
        
        # Try swap: remove one, add one
        for x in best_set[:]:
            remaining = [y for y in best_set if y != x]
            for z in nums:
                if z in best_set:
                    continue
                if _check_add(remaining, z):
                    new_set = remaining + [z]
                    new_val = _reciprocal_sum(new_set)
                    if new_val > best_val + 1e-12:
                        best_set = new_set
                        best_val = new_val
                        improved = True
        
        # Try remove one, add two
        for x in best_set[:]:
            remaining = [y for y in best_set if y != x]
            for z1 in nums:
                if z1 in best_set:
                    continue
                if not _check_add(remaining, z1):
                    continue
                temp = remaining + [z1]
                for z2 in nums:
                    if z2 in best_set or z2 == z1:
                        continue
                    if _check_add(temp, z2):
                        new_set = temp + [z2]
                        new_val = _reciprocal_sum(new_set)
                        if new_val > best_val + 1e-12:
                            best_set = new_set
                            best_val = new_val
                            improved = True
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    initial = _greedy_initial()
    
    if time.time() < deadline:
        result = _local_search(initial, deadline)
    else:
        result = initial
    
    return result
# EVOLVE-BLOCK-END
