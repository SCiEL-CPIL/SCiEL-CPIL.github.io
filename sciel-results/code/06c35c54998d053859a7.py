"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random

DEADLINE_SECONDS = 85.0

def _subset_sums(elements):
    """Compute all nonempty subset sums for a list of elements."""
    n = len(elements)
    if n == 0:
        return set()
    current = set()
    for e in elements:
        new_sums = set()
        for s in current:
            new_sums.add(s + e)
        new_sums.add(e)
        current = current | new_sums
    return current

def _check_no_divisibility(sums):
    """Check that no two distinct sums in the set divide each other."""
    s_list = sorted(sums)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            if b % a == 0:
                return False
    return True

def _is_valid(elements):
    """Check if the set of elements satisfies the divisibility constraint."""
    if len(elements) == 0:
        return True
    sums = _subset_sums(elements)
    return _check_no_divisibility(sums)

def _can_add(elements, new_elem, existing_sums):
    """Check if adding new_elem to elements maintains validity."""
    new_sums_set = set(existing_sums)
    new_sums_set.add(new_elem)
    for s in existing_sums:
        new_sums_set.add(s + new_elem)
    
    # Check pairs involving new sums
    new_only = {new_elem} | {s + new_elem for s in existing_sums}
    
    for ns in new_only:
        for other in new_sums_set:
            if ns == other:
                continue
            if other % ns == 0:
                return False
            if ns % other == 0:
                return False
    return True

def _greedy_from_order(order, deadline):
    """Greedily build a valid set by trying elements in the given order."""
    elements = []
    existing_sums = set()
    
    for elem in order:
        if time.time() > deadline:
            break
        if elem in elements:
            continue
        if _can_add(elements, elem, existing_sums):
            elements.append(elem)
            new_sums = set(existing_sums)
            new_sums.add(elem)
            for s in existing_sums:
                new_sums.add(s + elem)
            existing_sums = new_sums
    
    return elements

def _local_search(elements, pool, deadline, rng):
    """Local search with swaps and additions."""
    current = list(elements)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    rng.shuffle(available)
    
    existing_sums = _subset_sums(current)
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        for elem in available[:]:
            if time.time() > deadline:
                break
            if _can_add(current, elem, existing_sums):
                current.append(elem)
                current_set.add(elem)
                available.remove(elem)
                new_sums = set(existing_sums)
                new_sums.add(elem)
                for s in existing_sums:
                    new_sums.add(s + elem)
                existing_sums = new_sums
                improved = True
        
        # Try swaps
        if not improved and available:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                old_elem = current[i]
                for new_elem in available[:10]:
                    reduced = [x for j, x in enumerate(current) if j != i]
                    reduced_sums = _subset_sums(reduced)
                    if _can_add(reduced, new_elem, reduced_sums):
                        current[i] = new_elem
                        current_set.discard(old_elem)
                        current_set.add(new_elem)
                        available.remove(new_elem)
                        available.append(old_elem)
                        existing_sums = _subset_sums(current)
                        improved = True
                        break
                if improved:
                    break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best = []
    best_size = 0
    pool = list(range(1, 257))
    
    random.seed(42)
    
    # Run multiple random restarts
    for i in range(200):
        if time.time() > deadline - 5: # Leave 5s for final checks
            break
            
        order = pool[:]
        random.shuffle(order)
        
        result = _greedy_from_order(order, deadline - 5)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            
        # Local search on the best result found so far (or this one?)
        # Better to local search the best one found in this restart
        if len(result) > 0:
            improved = _local_search(result, pool, deadline - 5, random)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                
    # Final verification
    if best:
        if not _is_valid(best):
            best = [31, 32, 34]
            
    return [int(x) for x in best]
