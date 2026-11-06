"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _precompute_triples(N=180):
    """
    Precompute all forbidden triples (a, b, c) with distinct a, b, c in [1, N]
    such that 1/a = 1/b + 1/c.
    """
    triples_by_element = [set() for _ in range(N + 1)]
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if a >= 1 and a <= N and a != b and a != c:
                    triples_by_element[a].add((b, c))
                    triples_by_element[b].add((a, c))
                    triples_by_element[c].add((a, b))
    
    return triples_by_element

def _get_parity_partition(N=180):
    """
    Partition [1, N] into even and odd numbers.
    """
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    return evens, odds

def _max_valid_subset_in_parity(parity_class, triples_by_element, N=180):
    """
    Find maximum valid subset within a single parity class.
    Only consider triples where all three elements are in the same parity class.
    """
    parity_set = set(parity_class)
    
    # Build conflict graph: two elements conflict if they form a triple with a third in the same parity
    # Actually, we need: for each element x in parity_class, find all (y,z) in triples_by_element[x]
    # where y and z are also in parity_class
    
    current = set()
    remaining = list(parity_class)
    
    # Sort by degree (number of conflicts within parity class)
    def conflict_count(x, current_set):
        count = 0
        for (y, z) in triples_by_element[x]:
            if y in parity_set and z in parity_set:
                if y in current_set and z in current_set:
                    count += 1
        return count
    
    # Greedy: add elements with fewest conflicts first
    while remaining:
        best_x = None
        best_conflicts = float('inf')
        for x in remaining:
            conflicts = conflict_count(x, current)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_x = x
        
        if best_conflicts > 0:
            break
        
        current.add(best_x)
        remaining.remove(best_x)
    
    return current

def _is_valid_subset(subset, triples_by_element):
    """Check if subset has no forbidden triple."""
    subset_set = set(subset)
    for x in subset:
        for (y, z) in triples_by_element[x]:
            if y in subset_set and z in subset_set:
                return False
    return True

def _count_conflicts(x, current, triples_by_element):
    """Count how many conflicts x would have if added to current."""
    count = 0
    for (y, z) in triples_by_element[x]:
        if y in current and z in current:
            count += 1
    return count

def _greedy_with_conflict_heuristic(N=180, triples_by_element=None, start_from=None):
    """
    Greedy approach using conflict-count heuristic.
    If start_from is given, start with that set and add elements with fewest conflicts.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    if start_from is not None:
        current = set(start_from)
    else:
        current = set()
    
    # Build a list of candidates not in current
    candidates = [x for x in range(1, N + 1) if x not in current]
    
    # Sort by number of conflicts in the full set (as a proxy for "safe" elements)
    # Actually, let's just greedily add: at each step, pick the element with fewest conflicts
    while candidates:
        # Find element with fewest conflicts
        best_x = None
        best_conflicts = float('inf')
        for x in candidates:
            conflicts = _count_conflicts(x, current, triples_by_element)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_x = x
        
        if best_conflicts > 0:
            # No more elements can be added without conflict
            break
        
        current.add(best_x)
        candidates.remove(best_x)
    
    return current

def _greedy_descending(N=180, triples_by_element=None):
    """Greedy: include large numbers first."""
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set()
    for x in range(N, 0, -1):
        conflict = False
        for (y, z) in triples_by_element[x]:
            if y in current and z in current:
                conflict = True
                break
        if not conflict:
            current.add(x)
    
    return current

def _local_search_improve(subset, N=180, triples_by_element=None, max_iters=500, deadline=None):
    """
    Try to improve the subset by:
    1. Adding elements that don't create conflicts
    2. Swapping elements (remove one, add another)
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set(subset)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Phase 1: Try adding elements without removing
        for y in range(1, N + 1):
            if y in current:
                continue
            new_set = current | {y}
            if _is_valid_subset(new_set, triples_by_element):
                current = new_set
                improved = True
                break
        
        # Phase 2: Try swapping - remove one element, add another
        if not improved:
            for x in list(current):
                for y in range(1, N + 1):
                    if y in current:
                        continue
                    new_set = (current - {x}) | {y}
                    if _is_valid_subset(new_set, triples_by_element):
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
    
    return current

def _parity_based_construct(N=180, triples_by_element=None, deadline=None):
    """
    Use parity-based partition to decouple the search into two independent subproblems.
    Find max valid subset in evens and odds separately, then combine.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    evens, odds = _get_parity_partition(N)
    
    # Find max valid subset in each parity class
    even_subset = _max_valid_subset_in_parity(evens, triples_by_element, N)
    odd_subset = _max_valid_subset_in_parity(odds, triples_by_element, N)
    
    # Combine
    combined = even_subset | odd_subset
    
    # Verify validity of combined set
    if not _is_valid_subset(combined, triples_by_element):
        # Fallback: use greedy approach
        combined = set()
        for x in range(N, 0, -1):
            if _is_valid_subset(combined | {x}, triples_by_element):
                combined.add(x)
    
    return combined

def _local_search_with_removal(subset, N=180, triples_by_element=None, max_iters=200, deadline=None):
    """
    More aggressive local search: allow temporary removals to escape local optima.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set(subset)
    best = set(subset)
    iters = 0
    
    while iters < max_iters:
        if deadline and time.time() > deadline:
            break
        iters += 1
        improved = False
        
        # Try adding
        for y in range(1, N + 1):
            if y in current:
                continue
            new_set = current | {y}
            if _is_valid_subset(new_set, triples_by_element):
                current = new_set
                if len(current) > len(best):
                    best = set(current)
                improved = True
                break
        
        # Try swap
        if not improved:
            for x in list(current):
                for y in range(1, N + 1):
                    if y in current:
                        continue
                    new_set = (current - {x}) | {y}
                    if _is_valid_subset(new_set, triples_by_element):
                        current = new_set
                        if len(current) > len(best):
                            best = set(current)
                        improved = True
                        break
                if improved:
                    break
        
        # Try 2-swap: remove two, add two
        if not improved and len(current) >= 2:
            elements = list(current)
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    x1, x2 = elements[i], elements[j]
                    for y1 in range(1, N + 1):
                        if y1 in current:
                            continue
                        for y2 in range(1, N + 1):
                            if y2 in current or y2 == y1:
                                continue
                            new_set = (current - {x1, x2}) | {y1, y2}
                            if _is_valid_subset(new_set, triples_by_element):
                                current = new_set
                                if len(current) > len(best):
                                    best = set(current)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    N = 180
    triples_by_element = _precompute_triples(N)
    
    # Strategy 1: Greedy descending
    subset1 = _greedy_descending(N, triples_by_element)
    
    # Strategy 2: Start from upper half, add lower elements with conflict heuristic
    upper_half = set(range(91, 181))
    subset2 = _greedy_with_conflict_heuristic(N, triples_by_element, start_from=upper_half)
    
    # Strategy 3: Parity-based partition
    subset3 = _parity_based_construct(N, triples_by_element, deadline)
    
    # Pick the best initial solution
    candidates = [subset1, subset2, subset3]
    best_subset = max(candidates, key=len)
    
    # Local search improvement
    if time.time() < deadline:
        improved = _local_search_improve(best_subset, N, triples_by_element, max_iters=300, deadline=deadline)
        if len(improved) > len(best_subset):
            best_subset = improved
    
    # More aggressive local search if time permits
    if time.time() < deadline - 10:
        improved2 = _local_search_with_removal(best_subset, N, triples_by_element, max_iters=100, deadline=deadline)
        if len(improved2) > len(best_subset):
            best_subset = improved2
    
    # Final: try multiple restarts with random perturbations if time allows
    if time.time() < deadline - 5:
        best_size = len(best_subset)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Create a perturbed version
            perturbed = set(best_subset)
            # Remove 2-3 random elements
            elements_to_remove = random.sample(list(perturbed), min(3, len(perturbed)))
            for e in elements_to_remove:
                perturbed.discard(e)
            
            # Try to fill back up
            filled = _local_search_improve(perturbed, N, triples_by_element, max_iters=100, deadline=deadline)
            if len(filled) > best_size:
                best_subset = filled
                best_size = len(filled)
    
    # Convert to sorted list
    result = sorted(best_subset)
    
    return result
# EVOLVE-BLOCK-END
