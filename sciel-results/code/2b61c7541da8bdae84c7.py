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
    Faster check using bitsets for small element counts.
    For larger sets, use the set-based approach but with early termination.
    """
    n = len(elements)
    if n <= 1:
        return True
    
    max_sum = sum(elements)
    
    # Use sets of achievable sums per cardinality
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    # Track which sums have been seen in which cardinality for early conflict detection
    # sum_to_cardinals[s] = set of cardinalities that can achieve sum s
    sum_to_cards = {0: {0}}
    
    for elem in elements:
        # Build new entries
        new_entries = []
        for k in range(n, 0, -1):
            for s in list(sums_by_card[k - 1]):
                new_s = s + elem
                if new_s <= max_sum:
                    new_entries.append((k, new_s))
        
        for k, new_s in new_entries:
            if new_s not in sums_by_card[k]:
                sums_by_card[k].add(new_s)
                if new_s in sum_to_cards:
                    if k in sum_to_cards[new_s]:
                        continue
                    sum_to_cards[new_s].add(k)
                    if len(sum_to_cards[new_s]) > 1:
                        return False
                else:
                    sum_to_cards[new_s] = {k}
    
    return True


def _greedy_construct(max_value=256, max_size=60, deadline=None):
    """
    Greedy construction: try to add the smallest possible next element
    that maintains the cardinality-separation property.
    """
    elements = []
    used = set()
    
    for candidate in range(1, max_value + 1):
        if len(elements) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        if candidate in used:
            continue
        
        # Try adding candidate
        test_elements = elements + [candidate]
        if _is_cardinality_separated_fast(test_elements):
            elements.append(candidate)
            used.add(candidate)
    
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
    deadline = time.time() + 50  # Safety margin before 60s limit
    
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
    ]
    
    for start in heuristic_starts:
        if time.time() > deadline:
            break
        result = _try_heuristic_start(start, 256, 60, deadline)
        if len(result) > len(best):
            best = result
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
