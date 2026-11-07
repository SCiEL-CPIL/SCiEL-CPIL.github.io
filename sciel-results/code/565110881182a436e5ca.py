"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_divisibility(sums_set):
    """
    Check if any two distinct values in sums_set divide one another.
    sums_set is a set of integers.
    Returns True if conflict exists, False otherwise.
    """
    sorted_sums = sorted(sums_set)
    n = len(sorted_sums)
    max_sum = sorted_sums[-1]
    
    for i in range(n):
        si = sorted_sums[i]
        # Check if si divides any sj where sj > si
        # sj must be a multiple of si: 2*si, 3*si, ...
        # We can break early if 2*si > max_sum
        multiple = 2 * si
        while multiple <= max_sum:
            if multiple in sums_set:
                return True
            multiple += si
            
    return False

def _is_valid(arr):
    """
    Check if arr is a valid set.
    1. Distinct integers in 1..256.
    2. All non-empty subset sums are unique.
    3. No two distinct subset sums divide one another.
    """
    if not arr:
        return False
    
    # Check distinctness and range
    if len(set(arr)) != len(arr):
        return False
    for x in arr:
        if not (isinstance(x, int) and 1 <= x <= 256):
            return False
            
    n = len(arr)
    sums_set = set()
    
    # Generate all non-empty subset sums
    # Using bitmask enumeration
    for mask in range(1, 1 << n):
        s = 0
        # Optimization: iterate bits
        m = mask
        i = 0
        while m:
            if m & 1:
                s += arr[i]
            i += 1
            m >>= 1
            
        if s in sums_set:
            return False
        sums_set.add(s)
        
    # Check divisibility
    if _check_divisibility(sums_set):
        return False
        
    return True

def _try_greedy_extend(current, candidates, deadline):
    """
    Greedily extend current set with candidates.
    Returns the extended set.
    """
    current = list(current)
    used = set(current)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in used:
            continue
            
        trial = current + [cand]
        if _is_valid(trial):
            current = trial
            used.add(cand)
            
    return current

def construct():
    start_time = time.time()
    # Safety margin: return well before 90s. Let's aim for 85s.
    deadline = start_time + 85
    
    best_set = [31, 32, 34] # Known valid seed
    best_size = len(best_set)
    
    # Candidate pool: 1 to 256
    # Strategy: 
    # 1. Try greedy with descending order (largest first)
    # 2. Try greedy with ascending order
    # 3. Try randomized greedy with bias towards larger numbers
    # 4. Local search on best found
    
    rng = random.Random(42)
    
    # Pre-generate some candidate orderings
    # Ordering 1: Descending
    order_desc = list(range(256, 0, -1))
    # Ordering 2: Ascending
    order_asc = list(range(1, 257))
    # Ordering 3: Random, but weighted towards larger numbers? 
    # Actually, just random shuffles of the full range.
    
    # Phase 1: Deterministic Greedy
    if time.time() < deadline:
        current = _try_greedy_extend([], order_desc, deadline)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    if time.time() < deadline:
        current = _try_greedy_extend([], order_asc, deadline)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # Phase 2: Randomized Greedy Restarts
    # We want to explore different combinations.
    # Biasing towards larger numbers is good because they have fewer multiples.
    # Let's create a pool of "large" numbers and "small" numbers.
    
    large_nums = list(range(129, 257))
    small_nums = list(range(1, 129))
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Create a mixed candidate list
        # Try different mixes
        if attempt % 3 == 0:
            # Mostly large
            candidates = large_nums + small_nums
        elif attempt % 3 == 1:
            # Mostly small
            candidates = small_nums + large_nums
        else:
            # Random mix
            candidates = list(range(1, 257))
            
        rng.shuffle(candidates)
        
        current = _try_greedy_extend([], candidates, deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # Phase 3: Local Search
    # Try to improve best_set by swapping elements or adding/removing.
    # If we have a set of size K, can we find a set of size K+1?
    # Or can we find a "better" set of size K that allows growth?
    
    if best_size < 30 and time.time() < deadline:
        # Try to grow best_set further
        # Identify candidates not in best_set
        used_in_best = set(best_set)
        remaining_candidates = [x for x in range(1, 257) if x not in used_in_best]
        
        # Shuffle remaining
        rng.shuffle(remaining_candidates)
        
        # Try to add any of these
        improved = _try_greedy_extend(best_set, remaining_candidates, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
            
        # Try swapping one element for another to see if it allows growth
        # This is complex, so we'll do a simple random swap search
        for _ in range(500):
            if time.time() > deadline:
                break
            if not best_set:
                break
                
            # Pick a random index to swap
            idx = rng.randint(0, len(best_set) - 1)
            old_val = best_set[idx]
            
            # Pick a new value
            new_val = rng.randint(1, 256)
            if new_val in used_in_best:
                continue
                
            trial = best_set[:]
            trial[idx] = new_val
            trial.sort() # Keep sorted for consistency
            
            if _is_valid(trial):
                # Check if this swap allows us to add more elements
                # Try to extend this trial set
                used_in_trial = set(trial)
                remaining = [x for x in range(1, 257) if x not in used_in_trial]
                rng.shuffle(remaining)
                
                extended = _try_greedy_extend(trial, remaining, deadline)
                if len(extended) > best_size:
                    best_set = extended
                    best_size = len(best_set)
                elif len(extended) == best_size:
                    # Optional: keep the new set if it's "better" (e.g., larger elements?)
                    # For now, just keep the first one found of max size
                    pass

    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
