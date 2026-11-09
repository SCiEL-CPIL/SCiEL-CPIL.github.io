"""Rewritten program: greedy construction with exact cardinality-indexed subset-sum DP."""
import time
import random

# EVOLVE-BLOCK-START

def _is_cardinality_separated_fast(elements):
    """
    Faster check using bitsets (Python integers) for subset sums.
    B[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    Returns True if no sum appears in two different cardinality classes.
    """
    n = len(elements)
    if n <= 1:
        return True
    
    # B[k] is a bitmask of achievable sums for cardinality k
    B = [0] * (n + 1)
    B[0] = 1  # sum 0 is achievable with 0 elements
    
    for elem in elements:
        # Update backwards
        for k in range(n, 0, -1):
            # New sums for cardinality k come from sums of cardinality k-1 + elem
            B[k] |= (B[k - 1] << elem)
        
        # Check for collisions:
        # No sum should be present in two different B[k]
        seen = 0
        for k in range(n + 1):
            if B[k] & seen:
                return False
            seen |= B[k]
            
    return True


def _greedy_construct(elements, candidates, deadline):
    """
    Given an initial set of elements and a list of candidates, greedily add candidates.
    """
    current = list(elements)
    used = set(current)
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        
        test_elements = current + [candidate]
        if _is_cardinality_separated_fast(test_elements):
            current.append(candidate)
            used.add(candidate)
    
    return current


def _try_remove_and_add(elements, candidates, deadline):
    """
    Try to improve the set by removing one element and adding a different one,
    or just adding a new one.
    """
    best = list(elements)
    used = set(best)
    n = len(best)
    
    # First, try to add any candidate
    for candidate in candidates:
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        test_elements = best + [candidate]
        if _is_cardinality_separated_fast(test_elements):
            best.append(candidate)
            used.add(candidate)
    
    # Try swap: remove one element, add a candidate
    for i in range(len(best)):
        if time.time() > deadline:
            break
        # Remove best[i]
        removed = best[i]
        test_base = best[:i] + best[i+1:]
        used.remove(removed)
        
        for candidate in candidates:
            if time.time() > deadline:
                break
            if candidate in used:
                continue
            test_elements = test_base + [candidate]
            if _is_cardinality_separated_fast(test_elements):
                # Accept the swap
                best = test_elements
                used.add(candidate)
                break
        # Restore if no swap found
        if removed not in used:
            best = best[:i] + [removed] + best[i:]
            used.add(removed)
    
    return best


def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    """
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    best = []
    
    # Strategy 1: Greedy from smallest
    result1 = _greedy_construct([], list(range(1, 257)), deadline)
    if len(result1) > len(best):
        best = result1
    
    # Strategy 2: Greedy from largest
    result2 = _greedy_construct([], list(range(256, 0, -1)), deadline)
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
        list(range(1, 30, 2)),  # odd numbers
        list(range(2, 30, 2)),  # even numbers
    ]
    
    for start in heuristic_starts:
        if time.time() > deadline:
            break
        result = _greedy_construct(start, list(range(1, 257)), deadline)
        if len(result) > len(best):
            best = result
    
    # If we have time, try random restarts with greedy extension
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 30:
        attempts += 1
        # Pick a random subset of size 10 to start
        start_size = random.randint(5, 15)
        start = random.sample(range(1, 257), start_size)
        start.sort()
        result = _greedy_construct(start, list(range(1, 257)), deadline)
        if len(result) > len(best):
            best = result
    
    # Try local search improvement on the best solution
    if time.time() < deadline - 10:
        improved = _try_remove_and_add(best, list(range(1, 257)), deadline)
        if len(improved) > len(best):
            best = improved
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
