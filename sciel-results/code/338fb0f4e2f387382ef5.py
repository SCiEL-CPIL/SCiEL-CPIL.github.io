import time
import random
from itertools import combinations

def _subset_sums_set(elements):
    """Return the set of all non-empty subset sums."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_divisibility_free(sums):
    """Check no two distinct sums divide one another."""
    if len(sums) < 2:
        return True
    sorted_sums = sorted(sums)
    # For each sum, check if any multiple of it exists in the set
    sum_set = set(sums)
    for a in sorted_sums:
        # Check multiples: 2a, 3a, ... up to max
        max_val = sorted_sums[-1]
        for m in range(2 * a, max_val + 1, a):
            if m in sum_set:
                return False
    return True

def _is_valid(elements):
    """Full validation."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if not (1 <= x <= 256):
            return False
    sums = _subset_sums_set(elements)
    return _is_divisibility_free(sums)

def _check_add(elements, candidate, sums):
    """
    Check if adding candidate to elements keeps the set valid.
    Returns (new_sums, True) if valid, (None, False) otherwise.
    """
    if candidate in elements:
        return None, False
    new_sums = set(sums)
    for s in sums:
        new_sums.add(s + candidate)
    new_sums.add(candidate)
    if not _is_divisibility_free(new_sums):
        return None, False
    return new_sums, True

def _greedy_grow(elements, candidates, sums, deadline):
    """Greedily add candidates in given order."""
    elements = list(elements)
    sums = set(sums)
    for c in candidates:
        if time.time() > deadline:
            break
        if c in elements:
            continue
        new_sums, ok = _check_add(elements, c, sums)
        if ok:
            elements.append(c)
            sums = new_sums
    return elements, sums

def _backtrack(elements, sums, target_size, deadline, best_ref, candidate_pool):
    """
    Backtracking search to try to reach target_size.
    best_ref is a list [best_elements] that gets updated.
    """
    if len(elements) > len(best_ref[0]):
        best_ref[0] = list(elements)
    if len(elements) >= target_size:
        return
    if time.time() > deadline:
        return
    
    # Try adding each candidate not yet used
    for i, c in enumerate(candidate_pool):
        if time.time() > deadline:
            return
        if c in elements:
            continue
        new_sums, ok = _check_add(elements, c, sums)
        if ok:
            _backtrack(elements + [c], new_sums, target_size, deadline, best_ref, candidate_pool[i+1:])
            if time.time() > deadline:
                return

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    random.seed(42)
    all_nums = list(range(1, 257))
    
    best_result = [31, 32, 34]
    
    # Strategy 1: Start with a few well-chosen numbers and greedily grow
    # Key insight: numbers that are close together or in a "dense" region 
    # with large values tend to produce sums that are close together,
    # reducing the chance of divisibility.
    
    # Try various starting points - focus on larger numbers
    for base in range(200, 257):
        if time.time() > deadline:
            break
        start = [base]
        sums = _subset_sums_set(start)
        # Try adding nearby numbers first
        nearby = sorted(all_nums, key=lambda x: abs(x - base))
        nearby = [x for x in nearby if x != base]
        result, _ = _greedy_grow(start, nearby, sums, deadline)
        if len(result) > len(best_result):
            best_result = list(result)
    
    # Strategy 2: Start with pairs of nearby large numbers
    for a in range(180, 257, 3):
        if time.time() > deadline:
            break
        for b in range(a + 1, min(a + 30, 257)):
            if time.time() > deadline:
                break
            start = [a, b]
            sums = _subset_sums_set(start)
            nearby = sorted(all_nums, key=lambda x: min(abs(x - a), abs(x - b)))
            nearby = [x for x in nearby if x not in start]
            result, _ = _greedy_grow(start, nearby, sums, deadline)
            if len(result) > len(best_result):
                best_result = list(result)
    
    # Strategy 3: Randomized restarts with different orderings
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Random start size 1-4
        k = random.randint(1, 4)
        start = random.sample(all_nums, k)
        sums = _subset_sums_set(start)
        
        # Random ordering of remaining
        remaining = [x for x in all_nums if x not in start]
        random.shuffle(remaining)
        
        result, _ = _greedy_grow(start, remaining, sums, deadline)
        if len(result) > len(best_result):
            best_result = list(result)
    
    # Strategy 4: Local search / hill climbing on the best found
    if len(best_result) >= 3 and time.time() < deadline - 10:
        current = list(best_result)
        sums = _subset_sums_set(current)
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding one element
            used = set(current)
            candidates = [x for x in all_nums if x not in used]
            random.shuffle(candidates)
            
            for c in candidates:
                new_sums, ok = _check_add(current, c, sums)
                if ok:
                    current.append(c)
                    sums = new_sums
                    improved = True
                    break
            
            if not improved:
                # Try removing one and adding two different ones
                if len(current) >= 2:
                    remove_idx = random.randint(0, len(current) - 1)
                    removed = current.pop(remove_idx)
                    new_sums = _subset_sums_set(current)
                    
                    used = set(current)
                    candidates = [x for x in all_nums if x not in used]
                    random.shuffle(candidates)
                    
                    added_any = False
                    for c in candidates:
                        ns, ok = _check_add(current, c, new_sums)
                        if ok:
                            current.append(c)
                            new_sums = ns
                            added_any = True
                            break
                    
                    if not added_any:
                        current.append(removed)
                        new_sums = _subset_sums_set(current)
                    else:
                        sums = new_sums
                        if len(current) > len(best_result):
                            best_result = list(current)
    
    # Strategy 5: Targeted backtracking from promising starts
    if time.time() < deadline - 5:
        # Try to extend best_result with backtracking
        if len(best_result) < 10:
            target = len(best_result) + 1
            best_ref = [list(best_result)]
            used = set(best_result)
            pool = [x for x in all_nums if x not in used]
            # Sort by proximity to existing elements
            pool.sort(key=lambda x: min(abs(x - e) for e in best_result))
            sums = _subset_sums_set(best_result)
            _backtrack(list(best_result), sums, target, deadline, best_ref, pool)
            if len(best_ref[0]) > len(best_result):
                best_result = list(best_ref[0])
    
    # Final validation
    if _is_valid(best_result):
        return [int(x) for x in best_result]
    
    # Fallback
    return [31, 32, 34]
