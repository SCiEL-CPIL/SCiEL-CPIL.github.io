import numpy as np
import time
import random
from itertools import combinations

def _check_pair_compatible(a, b):
    """Check if two single elements are compatible (no divisibility between their subset sums {a}, {b}, {a+b})."""
    if a % b == 0 or b % a == 0:
        return False
    s = a + b
    if s % a == 0 or s % b == 0 or a % s == 0 or b % s == 0:
        return False
    return True

def _check_element_against_sums(x, sums_arr):
    """
    Check if element x can be added to a set with existing subset sums in sums_arr.
    Returns True if valid, False otherwise.
    """
    n = len(sums_arr)
    if n == 0:
        return True
    
    # x vs each existing sum
    if np.any(sums_arr % x == 0) or np.any(x % sums_arr == 0):
        return False
    
    # shifted sums
    shifted = sums_arr + x
    
    # shifted vs x
    if np.any(shifted % x == 0):
        return False
    
    # shifted vs sums_arr (both directions)
    chunk = 2000
    for i in range(0, n, chunk):
        e = min(i + chunk, n)
        if np.any(shifted[i:e, None] % sums_arr[None, :] == 0):
            return False
        if np.any(sums_arr[None, :] % shifted[i:e, None] == 0):
            return False
    
    # shifted vs shifted
    for i in range(0, n, chunk):
        e = min(i + chunk, n)
        rem = shifted[i:e, None] % shifted[None, :]
        # find zeros that are not self
        zeros = np.argwhere(rem == 0)
        for r, c in zeros:
            if i + r != c:
                return False
    
    return True

def _get_all_sums(elements):
    """Compute all non-empty subset sums for a list of elements."""
    sums = set()
    for i in range(1, len(elements) + 1):
        for combo in combinations(elements, i):
            sums.add(sum(combo))
    return sorted(sums)

def _is_valid_set(elements):
    """Check if a set of elements is fully valid."""
    sums = _get_all_sums(elements)
    n = len(sums)
    # Check all pairs for divisibility
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sums[i], sums[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _greedy_build(rng, deadline, compat_cache):
    """Greedy build with random restart ordering."""
    current = []
    current_sums = np.array([], dtype=np.int64)
    
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    
    for c in candidates:
        if time.time() > deadline:
            break
        if _check_element_against_sums(c, current_sums):
            shifted = current_sums + c if len(current_sums) > 0 else np.array([])
            if len(current_sums) > 0:
                new_sums = np.concatenate([current_sums, shifted, [c]])
            else:
                new_sums = np.array([c])
            new_sums = np.sort(new_sums)
            current_sums = new_sums
            current.append(c)
    
    return current

def _try_swap_improve(elements, deadline, max_iters=200):
    """Try to improve the set by swapping elements."""
    current = list(elements)
    current_sums = np.array(_get_all_sums(current), dtype=np.int64)
    improved = True
    iters = 0
    
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        # Try removing one element and adding a different one
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            
            # Remove element i
            remaining = current[:i] + current[i+1:]
            rem_sums = np.array(_get_all_sums(remaining), dtype=np.int64) if remaining else np.array([], dtype=np.int64)
            
            # Try to add each candidate not in current
            for c in range(256, 0, -1):
                if c in current:
                    continue
                if _check_element_against_sums(c, rem_sums):
                    # Valid swap
                    new_set = remaining + [c]
                    # Check if this helps (we'll validate later)
                    current = new_set
                    current_sums = np.array(_get_all_sums(current), dtype=np.int64)
                    improved = True
                    break
    
    return current

def _try_add_after_removal(elements, deadline):
    """Try to grow the set by removing one element and adding two."""
    current = list(elements)
    n = len(current)
    
    for i in range(n):
        if time.time() > deadline:
            return current
        
        remaining = current[:i] + current[i+1:]
        rem_sums = np.array(_get_all_sums(remaining), dtype=np.int64) if remaining else np.array([], dtype=np.int64)
        
        # Try to add one element
        for c in range(256, 0, -1):
            if c in current:
                continue
            if _check_element_against_sums(c, rem_sums):
                new_set = remaining + [c]
                new_sums = np.array(_get_all_sums(new_set), dtype=np.int64)
                
                # Try to add another
                for c2 in range(256, 0, -1):
                    if c2 in new_set:
                        continue
                    if _check_element_against_sums(c2, new_sums):
                        final_set = new_set + [c2]
                        if len(final_set) > len(current):
                            return final_set
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_set = []
    best_len = 0
    
    # Multiple random restarts
    for seed in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Greedy build
        current = _greedy_build(rng, deadline)
        
        if len(current) > best_len:
            best_set = current
            best_len = len(current)
        
        # Try swap improvement
        if time.time() < deadline - 5:
            improved = _try_swap_improve(current, deadline, max_iters=50)
            if len(improved) > best_len:
                best_set = improved
                best_len = len(improved)
        
        # Try add after removal
        if time.time() < deadline - 5:
            grown = _try_add_after_removal(improved, deadline)
            if len(grown) > best_len:
                best_set = grown
                best_len = len(grown)
    
    # Final validation
    if best_set:
        if not _is_valid_set(best_set):
            # Fallback: find a valid subset
            for size in range(len(best_set), 0, -1):
                for combo in combinations(best_set, size):
                    if _is_valid_set(list(combo)):
                        best_set = list(combo)
                        break
                else:
                    continue
                break
    
    # Ensure types are correct
    return [int(x) for x in best_set]
