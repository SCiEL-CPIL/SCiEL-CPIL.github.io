"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _subset_sums_distinct_fast(arr):
    """Faster check using incremental sums with set."""
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        if sums & new_sums:
            return False
        sums = sums | new_sums
    return True

def _check_valid_incremental(current_sums_set, new_elem):
    """Check if adding new_elem to current subset sums preserves distinctness."""
    for s in current_sums_set:
        if s + new_elem in current_sums_set:
            return False
    return True

def _validate(arr):
    """Validate: 12 distinct positive ints <= max, all subset sums distinct."""
    if len(arr) != 12:
        return False
    if len(set(arr)) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if any(x > 1048576 for x in arr):
        return False
    return _subset_sums_distinct_fast(arr)

def _score(arr):
    """Score is the largest element (lower is better)."""
    return max(arr)

def _greedy_construct(deadline=None):
    """Greedy construction: pick smallest valid element at each step."""
    if deadline is None:
        deadline = time.time() + 5
    
    n = 12
    elements = [1]
    subset_sums = [0, 1]
    subset_sums_set = set(subset_sums)
    
    for step in range(1, n):
        if time.time() > deadline:
            return None
        
        elem_set = set(elements)
        candidate = 1
        while True:
            if time.time() > deadline:
                return None
            if candidate not in elem_set:
                valid = True
                for s in subset_sums:
                    if s + candidate in subset_sums_set:
                        valid = False
                        break
                if valid:
                    elements.append(candidate)
                    subset_sums = subset_sums + [s + candidate for s in subset_sums]
                    subset_sums_set = set(subset_sums)
                    break
            candidate += 1
            if candidate > 1048576:
                return None
    
    return elements

def _local_search(arr, deadline):
    """Try to reduce elements while maintaining validity with smarter search."""
    arr = sorted(arr)
    n = len(arr)
    best_arr = arr[:]
    best_max = max(arr)
    
    # Multiple passes to find improvements
    for _pass in range(5):
        if time.time() > deadline:
            break
        improved = False
        # Try to reduce each element from largest to smallest
        for target_idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            current_val = arr[target_idx]
            arr_set = set(arr)
            
            # Try random candidates first, then systematic
            tried = set()
            # Try some random values
            for _ in range(50):
                if time.time() > deadline:
                    break
                new_val = random.randint(1, current_val - 1)
                if new_val in arr_set or new_val in tried:
                    continue
                tried.add(new_val)
                test_arr = arr[:]
                test_arr[target_idx] = new_val
                test_arr.sort()
                if _validate(test_arr):
                    arr = test_arr
                    arr_set = set(arr)
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best_arr = arr[:]
                    improved = True
                    break
            
            # If random didn't work, try systematic from current_val-1 down
            if not improved:
                for new_val in range(current_val - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    if new_val in arr_set or new_val in tried:
                        continue
                    test_arr = arr[:]
                    test_arr[target_idx] = new_val
                    test_arr.sort()
                    if _validate(test_arr):
                        arr = test_arr
                        arr_set = set(arr)
                        if max(arr) < best_max:
                            best_max = max(arr)
                            best_arr = arr[:]
                        improved = True
                        break
    
    return best_arr

def _randomized_construct(deadline=None):
    """Try randomized construction with different starting points."""
    if deadline is None:
        deadline = time.time() + 5
    
    n = 12
    best = None
    best_val = float('inf')
    
    # Try multiple random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        elements = []
        subset_sums_set = {0}
        
        # Start with a random small value
        start_val = random.randint(1, 3)
        elements.append(start_val)
        subset_sums_set = {0, start_val}
        
        success = True
        for step in range(1, n):
            if time.time() > deadline:
                success = False
                break
            
            elem_set = set(elements)
            found = False
            
            # Try candidates in randomized order around expected minimum
            base_candidate = sum(elements) + 1  # Upper bound for next valid element
            candidates = set()
            
            # Add systematic candidates
            for c in range(1, min(base_candidate + 10, 100)):
                if c not in elem_set:
                    candidates.add(c)
            
            # Add some random candidates
            for _ in range(20):
                c = random.randint(1, min(base_candidate, 200))
                if c not in elem_set:
                    candidates.add(c)
            
            for candidate in sorted(candidates):
                if time.time() > deadline:
                    success = False
                    break
                if candidate in elem_set:
                    continue
                
                valid = True
                for s in subset_sums_set:
                    if s + candidate in subset_sums_set:
                        valid = False
                        break
                
                if valid:
                    elements.append(candidate)
                    new_sums = [s + candidate for s in subset_sums_set]
                    subset_sums_set = subset_sums_set | set(new_sums)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success and len(elements) == n:
            val = max(elements)
            if val < best_val:
                best_val = val
                best = elements[:]
    
    return best

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget, 5s safety margin
    
    n = 12
    # Power of 2 solution as baseline
    best = [1 << i for i in range(n)]
    best_val = max(best)  # 2048
    
    # Try greedy construction
    if time.time() < deadline - 30:
        result = _greedy_construct(deadline - 30)
        if result is not None and _validate(result):
            val = max(result)
            if val < best_val:
                best_val = val
                best = result
    
    # Try randomized construction
    if time.time() < deadline - 20:
        result = _randomized_construct(deadline - 20)
        if result is not None and _validate(result):
            val = max(result)
            if val < best_val:
                best_val = val
                best = result
    
    # Local improvement pass
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline - 3)
        if _validate(improved) and max(improved) < best_val:
            best = improved
            best_val = max(best)
    
    # Final validation
    if _validate(best):
        return best
    
    # Fallback: return power of 2 solution
    return [1 << i for i in range(n)]
