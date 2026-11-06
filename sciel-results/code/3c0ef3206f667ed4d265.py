"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 80.0  # safety margin before 90s kill

def _subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(arr)
    sums = set()
    # Use bit manipulation to enumerate all subsets
    # For n=12, there are 4096 subsets
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _is_valid(arr):
    """Check validity: distinct positive integers, all subset sums distinct."""
    if len(set(arr)) != len(arr):
        return False
    if any(x <= 0 for x in arr):
        return False
    return _subset_sums_distinct(arr)

def _max_element(arr):
    return max(arr)

def _try_local_search(arr, max_steps=500000):
    """Try to improve by reducing the largest element via local moves."""
    arr = list(arr)
    n = len(arr)
    best_max = _max_element(arr)
    best_arr = list(arr)
    
    for step in range(max_steps):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        # Try reducing each element
        for i in range(n):
            old = arr[i]
            # Try reducing arr[i] by small amounts
            for delta in [1, 2, 3, 5, 7, 10, 15, 20]:
                new_val = old - delta
                if new_val <= 0:
                    continue
                if new_val in arr and new_val != old:
                    continue
                
                arr[i] = new_val
                if _is_valid(arr):
                    new_max = _max_element(arr)
                    if new_max < best_max:
                        best_max = new_max
                        best_arr = list(arr)
                        break
                arr[i] = old
        
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
    
    return best_arr

def _greedy_construct():
    """Greedy construction: build array one element at a time, choosing smallest valid."""
    arr = []
    n = 12
    
    for k in range(n):
        # Find smallest positive integer that can be added
        # such that all subset sums remain distinct
        # Start from 1 and go up
        found = False
        for candidate in range(1, 2048):
            test_arr = arr + [candidate]
            if len(set(test_arr)) != len(test_arr):
                continue
            if _subset_sums_distinct(test_arr):
                arr.append(candidate)
                found = True
                break
        if not found:
            # Fallback: use power of 2
            arr.append(1 << k)
    
    return arr

def _random_search(initial_arr, max_iters=200000):
    """Random perturbation search."""
    best_arr = list(initial_arr)
    best_max = _max_element(best_arr)
    n = len(best_arr)
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        candidate = list(best_arr)
        # Randomly perturb one or two elements
        num_perturb = rng.randint(1, 2)
        for _ in range(num_perturb):
            i = rng.randint(0, n - 1)
            # Try to reduce
            reduction = rng.randint(1, max(1, candidate[i] // 4))
            new_val = candidate[i] - reduction
            if new_val > 0 and new_val not in candidate:
                candidate[i] = new_val
        
        if _is_valid(candidate):
            new_max = _max_element(candidate)
            if new_max < best_max:
                best_max = new_max
                best_arr = candidate
    
    return best_arr

def _systematic_reduce_largest(arr):
    """Systematically try to reduce the largest element."""
    arr = list(arr)
    n = len(arr)
    best_arr = list(arr)
    best_max = _max_element(arr)
    
    # Sort to identify the largest
    # Try to reduce the largest element
    for _ in range(200):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        # Find index of max element
        max_idx = max(range(n), key=lambda i: arr[i])
        old_max = arr[max_idx]
        
        # Try reducing it
        for new_val in range(old_max - 1, 0, -1):
            if new_val in arr:
                continue
            arr[max_idx] = new_val
            if _is_valid(arr):
                best_arr = list(arr)
                best_max = new_val
                break
        else:
            arr[max_idx] = old_max
            break
        
        # Also try adjusting other elements to enable further reduction
        for i in range(n):
            if i == max_idx:
                continue
            for delta in range(1, 5):
                for sign in [-1, 1]:
                    new_val = arr[i] + sign * delta
                    if new_val <= 0:
                        continue
                    if new_val in arr:
                        continue
                    old = arr[i]
                    arr[i] = new_val
                    if _is_valid(arr):
                        # Now try to reduce max again
                        max_idx2 = max(range(n), key=lambda j: arr[j])
                        for nv in range(arr[max_idx2] - 1, 0, -1):
                            if nv in arr:
                                continue
                            arr[max_idx2] = nv
                            if _is_valid(arr):
                                break
                            arr[max_idx2] = arr[max_idx2] + 1  # restore
                        else:
                            arr[max_idx2] = arr[max_idx2]  # no change
                    else:
                        arr[i] = old
    
    return best_arr

def construct():
    """Construct 12 distinct positive integers with all subset sums distinct,
    minimizing the largest integer."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with powers of 2 (known valid)
    arr = [1 << i for i in range(12)]
    best_arr = list(arr)
    best_max = _max_element(arr)
    
    # Phase 1: Greedy construction
    greedy_arr = _greedy_construct()
    if _is_valid(greedy_arr):
        g_max = _max_element(greedy_arr)
        if g_max < best_max:
            best_max = g_max
            best_arr = list(greedy_arr)
    
    # Phase 2: Local search from best
    if time.time() - _START_TIME < _TIME_LIMIT - 5:
        local_arr = _try_local_search(best_arr)
        l_max = _max_element(local_arr)
        if l_max < best_max:
            best_max = l_max
            best_arr = local_arr
    
    # Phase 3: Random search
    if time.time() - _START_TIME < _TIME_LIMIT - 3:
        rand_arr = _random_search(best_arr)
        r_max = _max_element(rand_arr)
        if r_max < best_max:
            best_max = r_max
            best_arr = rand_arr
    
    # Phase 4: Systematic reduction
    if time.time() - _START_TIME < _TIME_LIMIT - 2:
        sys_arr = _systematic_reduce_largest(best_arr)
        s_max = _max_element(sys_arr)
        if s_max < best_max:
            best_max = s_max
            best_arr = sys_arr
    
    # Final validation
    if not _is_valid(best_arr):
        # Fallback to powers of 2
        return [1 << i for i in range(12)]
    
    return best_arr
# EVOLVE-BLOCK-END
