import random
import time

def _check_divisibility(sum_set):
    """Check that no two distinct values in sum_set divide each other."""
    sums = sorted(sum_set)
    n = len(sums)
    for i in range(n):
        si = sums[i]
        for j in range(i + 1, n):
            sj = sums[j]
            if sj % si == 0:
                return False
    return True

def _get_subset_sums(elements):
    """Get all non-empty subset sums as a set."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _can_add(elements, x, current_sums):
    """Check if x can be added to elements without violating divisibility."""
    # New sums are: current_sums ∪ {s + x for s in current_sums} ∪ {x}
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    
    # Check no two distinct values divide each other
    sums_list = sorted(new_sums)
    n = len(sums_list)
    for i in range(n):
        si = sums_list[i]
        for j in range(i + 1, n):
            sj = sums_list[j]
            if sj % si == 0:
                return False, None
    return True, new_sums

def _greedy_construct(pool, deadline, rng):
    """Greedy construction: add elements one by one if they don't violate constraints."""
    elements = []
    current_sums = set()
    
    for x in pool:
        if time.time() > deadline:
            break
        ok, new_sums = _can_add(elements, x, current_sums)
        if ok:
            elements.append(x)
            current_sums = new_sums
    
    return elements

def _greedy_with_backtrack(pool, max_size, deadline, rng):
    """Greedy with limited backtracking for better results."""
    best = []
    current_sums = set()
    
    def backtrack(elements, sums, idx):
        nonlocal best
        if time.time() > deadline:
            return
        if len(elements) > len(best):
            best = list(elements)
        if idx >= len(pool):
            return
        if len(elements) >= max_size:
            return
        
        x = pool[idx]
        # Try adding x
        ok, new_sums = _can_add(elements, x, sums)
        if ok:
            backtrack(elements + [x], new_sums, idx + 1)
        # Try skipping x (only if we haven't found max_size yet)
        if len(elements) < max_size:
            backtrack(elements, sums, idx + 1)
    
    backtrack([], set(), 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    best_result = [31, 32, 34]  # Fallback from seed
    
    rng = random.Random(42)
    
    # Strategy: Use numbers in upper range where subset sums are large
    # and less likely to have divisibility relationships
    # Numbers > 128: subset sums range from 129 to 256*30, but with few elements
    # Key: pick numbers such that no subset sum divides another
    
    # Approach: Try multiple random orderings of candidate pools
    # Focus on numbers in range [100, 256] where divisibility is less common
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Different strategies for different trials
        if trial < 50:
            # Strategy 1: Large numbers, random order
            pool = list(range(128, 257))
            rng.shuffle(pool)
            result = _greedy_construct(pool, deadline, rng)
        elif trial < 100:
            # Strategy 2: Numbers in [100, 256], random order
            pool = list(range(100, 257))
            rng.shuffle(pool)
            result = _greedy_construct(pool, deadline, rng)
        elif trial < 150:
            # Strategy 3: Mix of medium and large numbers
            pool = list(range(80, 257))
            rng.shuffle(pool)
            result = _greedy_construct(pool, deadline, rng)
        else:
            # Strategy 4: Use primes or numbers with few factors
            # Numbers that are close to each other in a high range
            pool = list(range(150, 257))
            rng.shuffle(pool)
            result = _greedy_construct(pool, deadline, rng)
        
        if len(result) > len(best_result):
            best_result = result
    
    # Additional targeted search: try to extend best_result
    if len(best_result) < 30 and time.time() < deadline - 5:
        # Try adding more elements to best_result
        current_elements = list(best_result)
        current_sums = _get_subset_sums(current_elements)
        
        # Try all remaining numbers
        for x in range(1, 257):
            if x in current_elements:
                continue
            if time.time() > deadline:
                break
            ok, new_sums = _can_add(current_elements, x, current_sums)
            if ok:
                current_elements.append(x)
                current_sums = new_sums
                if len(current_elements) > len(best_result):
                    best_result = current_elements
    
    # Ensure return type is correct
    result = [int(x) for x in best_result]
    # Verify distinctness and range
    if len(set(result)) != len(result):
        result = list(set(result))
    result = [x for x in result if 1 <= x <= 256]
    result = sorted(result)
    
    # Final validation
    if len(result) > 0:
        sums = _get_subset_sums(result)
        if not _check_divisibility(sums):
            # Fallback to known valid set
            result = [31, 32, 34]
    
    return result
