"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

MAX_SUM = 256 * 30
MAX_ELEMENTS = 30

def _get_subset_sums(elements):
    """Get all nonempty subset sums as a set."""
    sums = set()
    for e in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + e)
        sums = sums | new_sums | {e}
    return sums

def _is_divisibility_free(sums):
    """Check if no two distinct sums in the set divide each other."""
    sorted_sums = sorted(sums)
    n = len(sorted_sums)
    for i in range(n):
        a = sorted_sums[i]
        for j in range(i + 1, n):
            b = sorted_sums[j]
            if b % a == 0:
                return False
    return True

def _check_addition(current_sums, new_elem):
    """Check if adding new_elem maintains validity. Returns new sums or None."""
    new_sums = set(current_sums)
    new_sums.add(new_elem)
    for s in current_sums:
        new_sums.add(s + new_elem)
    if _is_divisibility_free(new_sums):
        return new_sums
    return None

def _build_conflict_graph():
    """
    Build conflict graph where edge (i,j) exists if i divides j or j divides i.
    This is a necessary condition: if two elements divide each other, they can't both be in a valid set.
    """
    n = 256
    conflict = [0] * (n + 1)  # bitmask of conflicting elements for each element
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and (i % j == 0 or j % i == 0):
                conflict[i] |= (1 << (j - 1))
    return conflict

def _branch_and_bound_max_independent_set(conflict, deadline, initial_set=None):
    """
    Find maximum independent set in conflict graph using branch and bound.
    Returns the set of element indices (1-indexed) in the independent set.
    """
    if initial_set is None:
        initial_set = set()
    
    # Order elements by degree (fewest conflicts first for better pruning)
    degrees = [bin(conflict[i]).count('1') for i in range(1, 257)]
    order = sorted(range(1, 257), key=lambda x: degrees[x - 1])
    
    best_set = set(initial_set)
    best_size = len(initial_set)
    
    def _branch(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: even if we add all remaining, can't beat best
        remaining = len(order) - idx
        if len(current_set) + remaining <= best_size:
            return
        
        if idx == len(order):
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            return
        
        elem = order[idx]
        
        # Option 1: Skip this element
        _branch(idx + 1, current_set, current_sums)
        
        # Option 2: Include this element (if no conflict with current set)
        conflicts_with_current = False
        for e in current_set:
            if conflict[elem] & (1 << (e - 1)):
                conflicts_with_current = True
                break
        
        if not conflicts_with_current:
            # Check if adding this element maintains subset sum validity
            new_sums = _check_addition(current_sums, elem)
            if new_sums is not None:
                _branch(idx + 1, current_set | {elem}, new_sums)
    
    # Start with empty set
    _branch(0, set(), set())
    
    return list(best_set)

def _greedy_grow(elements, deadline):
    """Greedily add elements that maintain validity."""
    current = list(elements)
    current_sums = _get_subset_sums(current)
    
    for _ in range(MAX_ELEMENTS - len(current)):
        if time.time() > deadline:
            break
        
        valid = []
        for val in range(1, 257):
            if val in current:
                continue
            new_sums = _check_addition(current_sums, val)
            if new_sums is not None:
                valid.append((val, new_sums))
        
        if not valid:
            break
        
        # Pick the one that results in fewest new sums (sparsest)
        val, new_sums = min(valid, key=lambda x: len(x[1]) - len(current_sums))
        current.append(val)
        current_sums = new_sums
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s limit
    
    # Build conflict graph
    conflict = _build_conflict_graph()
    
    best = []
    
    # Phase 1: Branch and bound to find large element-divisibility-free set
    if time.time() < deadline - 20:
        bnb_result = _branch_and_bound_max_independent_set(conflict, deadline - 15)
        if len(bnb_result) > len(best):
            best = bnb_result
    
    # Phase 2: Greedy growth from B&B result
    if len(best) > 0 and time.time() < deadline - 10:
        grown = _greedy_grow(best, deadline - 5)
        if len(grown) > len(best):
            best = grown
    
    # Phase 3: Multiple random restarts with greedy
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(attempt * 7919 + 13)
        
        # Random starting set
        start_size = random.randint(2, 5)
        start_elems = random.sample(range(1, 257), start_size)
        start_elems.sort()
        
        # Verify initial set is valid
        start_sums = _get_subset_sums(start_elems)
        if not _is_divisibility_free(start_sums):
            continue
        
        result = _greedy_grow(start_elems, deadline - 3)
        if len(result) > len(best):
            best = result
    
    # Phase 4: Local search - try swaps to improve
    if len(best) > 0 and time.time() < deadline - 3:
        current = list(best)
        current_sums = _get_subset_sums(current)
        
        for iteration in range(20):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for i, elem in enumerate(current):
                if time.time() > deadline - 2:
                    break
                
                reduced = current[:i] + current[i+1:]
                reduced_sums = _get_subset_sums(reduced)
                
                for new_val in range(1, 257):
                    if new_val in reduced:
                        continue
                    new_set = reduced + [new_val]
                    new_sums = _get_subset_sums(new_set)
                    
                    if _is_divisibility_free(new_sums):
                        trial = _greedy_grow(new_set, deadline - 2)
                        if len(trial) > len(best):
                            best = trial
                            current = trial[:]
                            current_sums = _get_subset_sums(trial)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
