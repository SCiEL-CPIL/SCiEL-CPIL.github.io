"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 85.0

def _all_subset_sums_fast(elements):
    """Generate all nonempty subset sums using iterative DP."""
    n = len(elements)
    if n == 0:
        return set()
    
    sums = set()
    sums.add(0)
    for e in elements:
        new_sums = set()
        for s in sums:
            new_s = s + e
            if new_s > 0:
                new_sums.add(new_s)
        sums.update(new_sums)
    sums.discard(0)
    return sums

def _is_valid(elements):
    """Check if no two distinct subset sums divide one another."""
    n = len(elements)
    if n == 0:
        return True
    
    sums = _all_subset_sums_fast(elements)
    sum_list = sorted(sums)
    m = len(sum_list)
    
    for i in range(m):
        a = sum_list[i]
        # Check if any larger sum is divisible by a
        for j in range(i+1, m):
            b = sum_list[j]
            if b % a == 0:
                return False
    return True

def _can_add(elements, candidate):
    """Check if adding candidate to elements keeps the set valid."""
    new_set = elements + [candidate]
    return _is_valid(new_set)

def _greedy_construct(start_elements, candidates, deadline):
    """Greedy: try to add candidates in order, keeping valid set."""
    current = list(start_elements)
    current_set = set(current)
    for c in candidates:
        if time.time() > deadline:
            break
        if c not in current_set:
            if _can_add(current, c):
                current.append(c)
                current_set.add(c)
    return current

def _local_search(elements, candidates, deadline, max_iter=200):
    """Local search: try adding, swapping to improve size."""
    current = list(elements)
    current_set = set(current)
    best = list(current)
    best_set = set(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try adding a random candidate from larger numbers first
        if candidates:
            # Bias toward larger numbers
            idx = random.randint(0, len(candidates) - 1)
            c = candidates[idx]
            if c not in current_set:
                if _can_add(current, c):
                    current.append(c)
                    current_set.add(c)
                    if len(current) > len(best):
                        best = list(current)
                        best_set = set(current)
        
        # Try removing a small element and adding a larger one
        if len(current) > 1 and candidates:
            # Remove a smaller element
            small_elements = [x for x in current if x < 128]
            if small_elements:
                remove_idx = current.index(random.choice(small_elements))
                removed = current.pop(remove_idx)
                current_set.discard(removed)
                
                # Try to add a larger candidate
                large_candidates = [x for x in candidates if x > removed and x not in current_set]
                if large_candidates:
                    c = random.choice(large_candidates)
                    if _can_add(current, c):
                        current.append(c)
                        current_set.add(c)
                    else:
                        current.append(removed)
                        current_set.add(removed)
                else:
                    current.append(removed)
                    current_set.add(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # All candidates from 1 to 256
    all_candidates = list(range(1, 257))
    
    # Strategy: use larger numbers to reduce subset sum collisions
    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Multiple restarts with different starting points
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Randomly select a starting set of larger numbers
        # Bias toward very large numbers
        if restart % 3 == 0:
            threshold = 150
        elif restart % 3 == 1:
            threshold = 120
        else:
            threshold = 100
            
        large_candidates = [x for x in all_candidates if x >= threshold]
        random.shuffle(large_candidates)
        
        # Start with 2-4 elements
        start_size = random.randint(2, 4)
        if len(large_candidates) < start_size:
            continue
        start_elements = large_candidates[:start_size]
        
        # Ensure start is valid
        if not _is_valid(start_elements):
            continue
        
        # Greedy add from remaining large numbers
        remaining_large = [x for x in large_candidates if x not in start_elements]
        random.shuffle(remaining_large)
        current = _greedy_construct(start_elements, remaining_large, deadline)
        
        # Local search with all remaining candidates
        all_remaining = [x for x in all_candidates if x not in current]
        random.shuffle(all_remaining)
        current = _local_search(current, all_remaining, deadline, max_iter=50)
        
        if len(current) > best_size:
            best_size = len(current)
            best_result = list(current)
        
        # Early termination if we hit a good size
        if best_size >= 12:
            break
    
    # Final verification
    if best_result and not _is_valid(best_result):
        # Fallback to a known small valid set
        best_result = [31, 32, 34]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
