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
    """Check if adding new_elem to elements maintains validity.
    existing_sums is the set of all nonempty subset sums of elements."""
    new_only = {new_elem} | {s + new_elem for s in existing_sums}
    all_sums = existing_sums | new_only
    
    for ns in new_only:
        for other in all_sums:
            if ns == other:
                continue
            if other % ns == 0:
                return False
            if ns % other == 0:
                return False
    return True

def _greedy_superincreasing(deadline):
    """Build a superincreasing-like set greedily, biased toward larger elements."""
    elements = []
    existing_sums = set()
    candidates = list(range(256, 0, -1))
    random.shuffle(candidates)
    
    for elem in candidates:
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

def _greedy_powers_of_2():
    """Return powers of 2 up to 128 (always valid)."""
    return [2**i for i in range(1, 9)]  # 2,4,8,...,128

def _local_search_with_removal(elements, pool, deadline, seed):
    """Local search that allows removals to escape local optima."""
    rng = random.Random(seed)
    current = list(elements)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    rng.shuffle(available)
    
    existing_sums = _subset_sums(current)
    best_current = list(current)
    best_size = len(current)
    
    max_iterations = 50
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try to add
        added = False
        for elem in available[:30]:
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
                added = True
                if len(current) > best_size:
                    best_size = len(current)
                    best_current = list(current)
                break
        
        if not added:
            # Try removal + addition (downhill move)
            removed = False
            for idx in range(len(current)):
                if time.time() > deadline:
                    break
                old_elem = current[idx]
                reduced = [x for i, x in enumerate(current) if i != idx]
                reduced_sums = _subset_sums(reduced)
                
                # Try to add something from available
                found = False
                for new_elem in available[:20]:
                    if time.time() > deadline:
                        break
                    if _can_add(reduced, new_elem, reduced_sums):
                        # Perform removal and addition
                        current = reduced + [new_elem]
                        current_set.discard(old_elem)
                        current_set.add(new_elem)
                        available.remove(new_elem)
                        available.append(old_elem)
                        rng.shuffle(available)
                        existing_sums = _subset_sums(current)
                        removed = True
                        found = True
                        break
                
                if found:
                    break
            
            if not removed:
                break
    
    return best_current

def _try_randomized_greedy(deadline, seed, bias='random'):
    """Randomized greedy with different biases."""
    rng = random.Random(seed)
    elements = []
    existing_sums = set()
    
    if bias == 'descending':
        candidates = list(range(256, 0, -1))
    elif bias == 'ascending':
        candidates = list(range(1, 257))
    elif bias == 'powers':
        candidates = [2**i for i in range(1, 9)] + list(range(1, 257))
    else:
        candidates = list(range(1, 257))
    
    rng.shuffle(candidates)
    
    for elem in candidates:
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

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best = []
    best_size = 0
    
    # Start with known valid set
    base_set = _greedy_powers_of_2()
    best = list(base_set)
    best_size = len(best)
    
    # Try multiple greedy strategies
    strategies = [
        ('descending', 1),
        ('ascending', 2),
        ('random', 3),
        ('random', 4),
        ('random', 5),
        ('random', 6),
        ('random', 7),
        ('random', 8),
    ]
    
    for bias, seed in strategies:
        if time.time() > deadline:
            break
        result = _try_randomized_greedy(deadline, seed, bias)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Local search with removals on best found
    if best_size > 0 and time.time() < deadline:
        pool = list(range(1, 257))
        for seed in range(10):
            if time.time() > deadline:
                break
            improved = _local_search_with_removal(best, pool, deadline, seed)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Try superincreasing greedy
    if time.time() < deadline:
        result = _greedy_superincreasing(deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Final verification
    if best:
        if not _is_valid(best):
            best = base_set
    
    return [int(x) for x in best]
