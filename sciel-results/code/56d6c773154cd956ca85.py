"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _subset_sums_distinct(arr):
    """Check if all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    # Use iterative subset sum generation
    # sums[k] for k in 0..n-1 tracks sums of subsets of arr[0..k]
    current = {0}
    for a in arr:
        new = set()
        for s in current:
            new.add(s + a)
        current = current | new
        if len(current) != 2 ** (arr.index(a) + 1) if False else len(current) != (2 ** (len([x for x in arr[:arr.index(a)+1]]))):
            pass
    # Simpler: just generate all subset sums
    sums = set()
    n = len(arr)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _subset_sums_set(arr):
    """Return the set of all subset sums."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        sums.add(s)
    return sums

def _is_valid(arr):
    """Check distinctness, positivity, and bounds."""
    if len(arr) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if any(x > 1048576 for x in arr):
        return False
    if len(set(arr)) != 12:
        return False
    return _subset_sums_distinct(arr)

def _greedy_construct(seed=0):
    """Greedy: pick smallest valid next element at each step."""
    rng = random.Random(seed)
    arr = []
    # Start with 1
    arr.append(1)
    current_sums = {0, 1}
    
    for i in range(1, 12):
        # Find smallest a such that adding a keeps all subset sums distinct
        # New sums = old_sums ∪ (old_sums + a)
        # Need: (old_sums + a) ∩ old_sums = empty
        # i.e., a must not equal any difference of two subset sums
        # Equivalently: for all s in current_sums, s + a not in current_sums
        # i.e., a not in {s2 - s1 : s1, s2 in current_sums, s2 > s1}
        
        # The forbidden values are all positive differences between subset sums
        forbidden = set()
        sums_list = sorted(current_sums)
        for j in range(len(sums_list)):
            for k in range(j+1, len(sums_list)):
                d = sums_list[k] - sums_list[j]
                if d > 0:
                    forbidden.add(d)
        
        # Find smallest positive integer not in forbidden and not in arr
        a = 1
        while a in forbidden or a in arr:
            a += 1
        
        arr.append(a)
        # Update current_sums
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + a)
        current_sums = current_sums | new_sums
    
    return arr

def _local_search(arr, max_time=5.0):
    """Try to reduce the largest element via local moves."""
    start = time.time()
    best = list(arr)
    best_max = max(best)
    
    # Sort for consistency
    best.sort()
    
    while time.time() - start < max_time:
        improved = False
        # Try to reduce each element
        for i in range(len(best)):
            original = best[i]
            # Try reducing best[i]
            for delta in range(1, min(original, 200) + 1):
                candidate = best[:]
                candidate[i] = original - delta
                if candidate[i] <= 0:
                    break
                if len(set(candidate)) != 12:
                    continue
                if _subset_sums_distinct(candidate):
                    best = candidate
                    best_max = max(best)
                    improved = True
                    break
        if not improved:
            break
    
    return best, best_max

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Phase 1: Greedy construction
    best_arr = _greedy_construct(0)
    best_max = max(best_arr)
    
    # Phase 2: Local search to improve
    best_arr, best_max = _local_search(best_arr, max_time=10.0)
    
    # Phase 3: Try different greedy strategies
    # Strategy: sometimes pick a larger element early to create gaps
    for seed in range(1, 50):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(seed)
        arr = []
        arr.append(1)
        current_sums = {0, 1}
        
        valid = True
        for i in range(1, 12):
            forbidden = set()
            sums_list = sorted(current_sums)
            for j in range(len(sums_list)):
                for k in range(j+1, len(sums_list)):
                    d = sums_list[k] - sums_list[j]
                    if d > 0:
                        forbidden.add(d)
            
            # With some probability, pick a slightly larger value to create structure
            a = 1
            while a in forbidden or a in arr:
                a += 1
            
            # Occasionally jump ahead
            if rng.random() < 0.3 and i < 10:
                jump = a + rng.randint(1, 3)
                # Check if jump is valid
                if jump not in forbidden and jump not in arr:
                    a = jump
            
            arr.append(a)
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + a)
            current_sums = current_sums | new_sums
        
        if _subset_sums_distinct(arr):
            arr.sort()
            if max(arr) < best_max:
                best_max = max(arr)
                best_arr = arr
                # Local search
                best_arr, best_max = _local_search(best_arr, max_time=3.0)
    
    # Phase 4: More aggressive local search on best
    if time.time() < deadline - 5:
        best_arr, best_max = _local_search(best_arr, max_time=15.0)
    
    return best_arr
# EVOLVE-BLOCK-END
