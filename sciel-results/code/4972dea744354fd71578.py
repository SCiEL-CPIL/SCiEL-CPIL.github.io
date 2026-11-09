"""Find maximum cardinality dissociated reciprocal subset of [1,22]."""
import time
from fractions import Fraction
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def _mod_subset_sums_distinct(elements, mod):
    """Fast check: are all subset sums of reciprocals distinct mod p?
    Returns True if no collision found mod p (necessary condition for exact distinctness)."""
    n = len(elements)
    if n == 0:
        return True
    # Compute 1/e mod p for each element
    recips = [pow(e, mod - 2, mod) for e in elements]
    # Generate subset sums incrementally
    sums = [0]
    for r in recips:
        new_sums = [(s + r) % mod for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _exact_subset_sums_distinct(elements):
    """Exact check using Fraction arithmetic."""
    n = len(elements)
    if n == 0:
        return True
    recips = [Fraction(1, e) for e in elements]
    sums = [Fraction(0, 1)]
    for r in recips:
        new_sums = [s + r for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _is_dissociated(elements):
    """Check if all subset sums of reciprocals are distinct (exact)."""
    return _exact_subset_sums_distinct(elements)

def _is_dissociated_fast(elements):
    """Two-tier check: fast modular filter then exact."""
    if not _mod_subset_sums_distinct(elements, 10**9 + 7):
        return False
    return _exact_subset_sums_distinct(elements)

def _try_add(element, current_elements):
    """Check if adding element to current_elements keeps dissociation."""
    return _is_dissociated_fast(current_elements + [element])

def _backtrack_search(elements_pool, current_set, best_set, deadline, max_size_target):
    """Backtracking search to find largest dissociated set."""
    if time.time() > deadline:
        return best_set
    if len(current_set) > len(best_set):
        best_set = list(current_set)
    if len(current_set) >= max_size_target:
        return best_set
    
    for i, elem in enumerate(elements_pool):
        if elem in current_set:
            continue
        new_set = current_set + [elem]
        if _is_dissociated_fast(new_set):
            best_set = _backtrack_search(elements_pool[i+1:], new_set, best_set, deadline, max_size_target)
            if time.time() > deadline:
                return best_set
            if len(best_set) >= max_size_target:
                return best_set
    return best_set

def _find_set_of_size(target_size, deadline):
    """Find a dissociated set of exactly target_size elements from [1,22]."""
    
    def backtrack(remaining_pool, current):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            if _is_dissociated_fast(current):
                return list(current)
            return None
        
        if len(remaining_pool) < target_size - len(current):
            return None
        
        for i, elem in enumerate(remaining_pool):
            if time.time() > deadline:
                return None
            new_current = current + [elem]
            if _is_dissociated_fast(new_current):
                result = backtrack(remaining_pool[i+1:], new_current)
                if result is not None:
                    return result
        return None
    
    pool = list(range(22, 0, -1))
    return backtrack(pool, [])

def _greedy_construct(pool, deadline):
    """Greedy: add elements one by one if they maintain dissociation."""
    current = []
    for elem in pool:
        if time.time() > deadline:
            break
        if _is_dissociated_fast(current + [elem]):
            current.append(elem)
    return current

def _randomized_greedy(pool, deadline, n_restarts=50):
    """Multiple randomized greedy attempts."""
    best = []
    for _ in range(n_restarts):
        if time.time() > deadline:
            break
        shuffled = list(pool)
        random.shuffle(shuffled)
        result = _greedy_construct(shuffled, deadline)
        if len(result) > len(best):
            best = result
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # Known valid 9-element set
    seed_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = list(seed_set)
    
    all_elements = list(range(1, 23))
    
    # Phase 1: Greedy from seed with different orderings
    remaining = [i for i in range(1, 23) if i not in seed_set]
    
    # Try greedy adding remaining elements in various orders
    for order in [remaining, sorted(remaining, reverse=True), sorted(remaining)]:
        if time.time() > deadline:
            break
        current = list(seed_set)
        for elem in order:
            if time.time() > deadline:
                break
            if _is_dissociated_fast(current + [elem]):
                current.append(elem)
        if len(current) > len(best):
            best = list(current)
    
    # Phase 2: Randomized greedy with multiple restarts
    if time.time() < deadline:
        rand_best = _randomized_greedy(all_elements, deadline, n_restarts=100)
        if len(rand_best) > len(best):
            best = rand_best
    
    # Phase 3: Try to find larger sets via targeted backtracking
    target_size = len(best) + 1
    while target_size <= 14 and time.time() < deadline:
        found = _find_set_of_size(target_size, deadline)
        if found is not None:
            best = found
            target_size += 1
        else:
            break
    
    # Phase 4: If we still have time, try more randomized searches
    if time.time() < deadline and len(best) < 12:
        more_best = _randomized_greedy(all_elements, deadline, n_restarts=200)
        if len(more_best) > len(best):
            best = more_best
    
    return best

# EVOLVE-BLOCK-END
