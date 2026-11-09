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


def _is_cardinality_separated_fast(elements):
    """
    Faster check using bitsets (Python integers) for subset sums.
    B[k] is an integer where bit s is 1 if sum s is achievable with k elements.
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
            # Shift B[k-1] by elem and OR with existing B[k]
            B[k] |= (B[k - 1] << elem)
        
        # Check for collisions:
        # No sum should be present in two different B[k]
        seen = 0
        for k in range(n + 1):
            if B[k] & seen:
                return False
            seen |= B[k]
            
    return True


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
        test_elements = elements + [current_candidate]
        if _is_cardinality_separated_fast(test_elements):
            elements.append(current_candidate)
            used.add(current_candidate)
            current_candidate += 1
        else:
            # If adding this candidate fails, we could try to skip it.
            # But if we are stuck (no candidate works), we might need to backtrack.
            # Simple greedy just skips.
            # To improve, we can try to see if any LARGER candidate works.
            # The loop naturally does this by incrementing current_candidate.
            pass
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
        
        test_elements = elements + [candidate]
        if _is_cardinality_separated_fast(test_elements):
            elements.append(candidate)
            used.add(candidate)
    
    return elements


def _try_heuristic_start(start_range, max_value=256, max_size=60, deadline=None):
    """
    Try starting with a heuristic initial set, then greedily extend.
    """
    elements = list(start_range)
    used = set(elements)
    
    for candidate in range(1, max_value + 1):
        if len(elements) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        if candidate in used:
            continue
        
        test_elements = elements + [candidate]
        if _is_cardinality_separated_fast(test_elements):
            elements.append(candidate)
            used.add(candidate)
    
    return elements


def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    """
    deadline = time.time() + 55  # Safety margin before 60s limit
    
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
    ]
    
    for start in heuristic_starts:
        if time.time() > deadline:
            break
        result = _try_heuristic_start(start, 256, 60, deadline)
        if len(result) > len(best):
            best = result
            
    # If we have time, try random restarts with greedy extension
    import random
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 20:
        attempts += 1
        # Pick a random subset of size 10 to start
        start_size = random.randint(5, 15)
        start = random.sample(range(1, 257), start_size)
        start.sort()
        result = _try_heuristic_start(start, 256, 60, deadline)
        if len(result) > len(best):
            best = result

    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
