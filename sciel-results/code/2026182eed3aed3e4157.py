"""Rewritten program: greedy construction with exact cardinality-indexed subset-sum DP."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_cardinality_separated(elements):
    """
    Check if a set of distinct positive integers satisfies the property:
    whenever two subsets have different cardinalities, their sums must differ.
    
    We track, for each cardinality k, the set of all achievable sums.
    Then we verify that no sum appears in two different cardinality classes.
    """
    n = len(elements)
    max_sum = sum(elements)
    
    # sums_by_card[k] = set of achievable sums using exactly k elements
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for elem in elements:
        # Iterate backwards to avoid using the same element twice
        for k in range(n, 0, -1):
            for s in list(sums_by_card[k - 1]):
                new_s = s + elem
                if new_s <= max_sum:
                    sums_by_card[k].add(new_s)
    
    # Check pairwise disjointness across different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_card[i] & sums_by_card[j]:
                return False
    return True


def _build_bitmasks(elements):
    """
    Build B[k] bitmasks: B[k] has bit s set if sum s is achievable with exactly k elements.
    Returns list of bitmasks, or None if collision detected.
    """
    n = len(elements)
    if n <= 0:
        return [1]
    
    B = [0] * (n + 1)
    B[0] = 1  # sum 0 is achievable with 0 elements
    
    for elem in elements:
        # Update backwards
        for k in range(n, 0, -1):
            B[k] |= (B[k - 1] << elem)
        
        # Check for collisions across different cardinalities
        seen = 0
        for k in range(n + 1):
            if B[k] & seen:
                return None
            seen |= B[k]
            
    return B


def _is_cardinality_separated_fast(elements):
    """
    Faster check using bitsets (Python integers) for subset sums.
    """
    return _build_bitmasks(elements) is not None


def _check_addition(elements, new_elem):
    """
    Check if adding new_elem to elements preserves the property.
    Returns the updated B list if valid, else None.
    This is faster than rebuilding from scratch since we only need to check
    the new contributions.
    """
    n = len(elements)
    new_n = n + 1
    
    # First build the existing B
    B = [0] * (n + 1)
    B[0] = 1
    
    for elem in elements:
        for k in range(n, 0, -1):
            B[k] |= (B[k - 1] << elem)
    
    # Now add new_elem
    new_B = [0] * (new_n + 1)
    for k in range(n + 1):
        new_B[k] = B[k]
    
    # Add contributions from new_elem
    for k in range(new_n, 0, -1):
        if k - 1 <= n:
            new_B[k] |= (B[k - 1] << new_elem)
    
    # Check for collisions
    seen = 0
    for k in range(new_n + 1):
        if new_B[k] & seen:
            return None
        seen |= new_B[k]
    
    return new_B


def _greedy_construct(max_value=256, max_size=60, deadline=None):
    """
    Greedy construction with backtracking on failure.
    """
    elements = []
    used = set()
    current_candidate = 1
    
    while len(elements) < max_size and current_candidate <= max_value:
        if deadline is not None and time.time() > deadline:
            break
        
        if current_candidate in used:
            current_candidate += 1
            continue
            
        # Try adding current_candidate
        if _check_addition(elements, current_candidate) is not None:
            elements.append(current_candidate)
            used.add(current_candidate)
            current_candidate += 1
        else:
            current_candidate += 1
            
    return elements


def _greedy_construct_reverse(max_value=256, max_size=60, deadline=None):
    """
    Try building from larger values down, or use a different ordering.
    """
    elements = []
    used = set()
    
    for candidate in range(max_value, 0, -1):
        if len(elements) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        if candidate in used:
            continue
        
        if _check_addition(elements, candidate) is not None:
            elements.append(candidate)
            used.add(candidate)
    
    return elements


def _try_heuristic_start(start_range, max_value=256, max_size=60, deadline=None):
    """
    Try starting with a heuristic initial set, then greedily extend.
    """
    elements = list(start_range)
    used = set(elements)
    
    # Verify the start is valid
    if not _is_cardinality_separated_fast(elements):
        return []
    
    for candidate in range(1, max_value + 1):
        if len(elements) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        if candidate in used:
            continue
        
        if _check_addition(elements, candidate) is not None:
            elements.append(candidate)
            used.add(candidate)
    
    return elements


def _local_search_improve(elements, max_value=256, max_size=60, deadline=None):
    """
    Local search: try swapping each element with a smaller unused value to free up space,
    then greedily extend.
    """
    if not elements:
        return elements
    
    best = list(elements)
    used = set(best)
    
    # Try each position for a swap
    for i in range(len(best)):
        if deadline is not None and time.time() > deadline:
            break
        
        orig_val = best[i]
        
        # Try replacing with smaller unused values
        for new_val in range(1, orig_val):
            if new_val in used:
                continue
            
            # Test the swap
            test_elements = best[:i] + [new_val] + best[i+1:]
            if _is_cardinality_separated_fast(test_elements):
                # Try to extend greedily
                test_used = set(test_elements)
                improved = list(test_elements)
                
                for candidate in range(1, max_value + 1):
                    if len(improved) >= max_size:
                        break
                    if deadline is not None and time.time() > deadline:
                        break
                    if candidate in test_used:
                        continue
                    if _check_addition(improved, candidate) is not None:
                        improved.append(candidate)
                        test_used.add(candidate)
                
                if len(improved) > len(best):
                    best = improved
                    used = test_used
                    break  # Restart with the improved set
        
        # Also try removing an element and re-extending
        if deadline is not None and time.time() > deadline:
            break
        
        reduced = best[:i] + best[i+1:]
        if _is_cardinality_separated_fast(reduced):
            reduced_used = set(reduced)
            improved = list(reduced)
            
            for candidate in range(1, max_value + 1):
                if len(improved) >= max_size:
                    break
                if deadline is not None and time.time() > deadline:
                    break
                if candidate in reduced_used:
                    continue
                if _check_addition(improved, candidate) is not None:
                    improved.append(candidate)
                    reduced_used.add(candidate)
            
            if len(improved) > len(best):
                best = improved
                used = reduced_used
    
    return best


def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best = []
    
    # Strategy 1: Greedy from smallest
    result1 = _greedy_construct(256, 60, deadline)
    if len(result1) > len(best):
        best = result1
    
    # Strategy 2: Greedy from largest
    result2 = _greedy_construct_reverse(256, 60, deadline)
    if len(result2) > len(best):
        best = result2
    
    # Strategy 3: Try some heuristic starts
    heuristic_starts = [
        list(range(1, 10)),
        list(range(1, 15)),
        [1, 2, 4, 8, 16, 32, 64, 128],  # powers of 2
        list(range(2, 20)),
        list(range(10, 40)),
        list(range(50, 80)),
        list(range(1, 25, 2)),  # odd numbers
        list(range(2, 50, 4)),  # multiples of 4
    ]
    
    for start in heuristic_starts:
        if time.time() > deadline:
            break
        result = _try_heuristic_start(start, 256, 60, deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Local search improvement on best result
    if best and time.time() < deadline - 5:
        improved = _local_search_improve(best, 256, 60, deadline)
        if len(improved) > len(best):
            best = improved
    
    # If we have time, try random restarts with greedy extension
    import random
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10 and attempts < 30:
        attempts += 1
        # Pick a random subset of size 10 to start
        start_size = random.randint(5, 15)
        start = random.sample(range(1, 257), start_size)
        start.sort()
        result = _try_heuristic_start(start, 256, 60, deadline)
        if len(result) > len(best):
            best = result
            # Apply local search to new best
            if time.time() < deadline - 10:
                improved = _local_search_improve(best, 256, 60, deadline)
                if len(improved) > len(best):
                    best = improved

    # Final local search pass if time permits
    if best and time.time() < deadline - 2:
        improved = _local_search_improve(best, 256, 60, deadline)
        if len(improved) > len(best):
            best = improved

    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
