"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _check_property(elements):
    """Check if subset sums with different cardinalities are all distinct."""
    n = len(elements)
    # sum_by_card[k] = set of subset sums of size k
    sum_by_card = [set() for _ in range(n + 1)]
    sum_by_card[0].add(0)
    
    for i, e in enumerate(elements):
        # Process in reverse to avoid using e multiple times
        for k in range(i, -1, -1):
            for s in sum_by_card[k]:
                sum_by_card[k + 1].add(s + e)
    
    # Check no sum appears in two different cardinality sets
    seen = {}
    for k in range(n + 1):
        for s in sum_by_card[k]:
            if s in seen:
                if seen[s] != k:
                    return False
            else:
                seen[s] = k
    return True

def _try_add(elements, candidate, sum_by_card, n_current):
    """Try adding candidate, return new sum_by_card or None if invalid."""
    n = n_current
    new_sum_by_card = [set() for _ in range(n + 2)]
    # Copy existing
    for k in range(n + 1):
        new_sum_by_card[k] = set(sum_by_card[k])
    # Add subsets including candidate
    for k in range(n + 1):
        for s in sum_by_card[k]:
            new_sum_by_card[k + 1].add(s + candidate)
    
    # Check property
    seen = {}
    for k in range(n + 2):
        for s in new_sum_by_card[k]:
            if s in seen:
                if seen[s] != k:
                    return None
            else:
                seen[s] = k
    return new_sum_by_card

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Greedy approach: try to build a set by adding elements
    # Start with small elements and grow
    # Use backtracking with heuristic ordering
    
    # First, try a simple constructive approach
    # Elements where each is > sum of previous ensures all subset sums distinct
    # But that's too restrictive. Let's search.
    
    # Strategy: try different starting sets and greedily extend
    # Or use a smarter search
    
    # Let's try: sort candidates, use backtracking
    # For each candidate from 256 down to 1, decide include/exclude
    # But 256 candidates is too many. Use a heuristic.
    
    # Better: start with a known good small set, then try to add more
    # Or: generate candidates and use greedy with local search
    
    # Simple approach: try all subsets of a smaller candidate pool
    # Let's use candidates 1..60 or so, and search for max valid subset
    
    # Actually, let's try a direct constructive method first
    # a_i = 2^(i-1) gives 8 elements: 1,2,4,8,16,32,64,128
    # Can we do better?
    
    # Let's try: 1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 250
    # Check if this works...
    
    # Instead, let's do a proper search
    # Use the fact that if elements are "sufficiently spread out" it works
    
    # Heuristic: try to build a set where consecutive elements differ by at least the sum of all smaller elements
    # That's the superincreasing condition, too strict
    
    # Alternative: use a_1=1, a_2=2, a_3=4, ..., then try to add larger elements
    # that don't create conflicts
    
    # Let me implement a backtracking search on a reduced candidate set
    
    # Candidate pool: use numbers that are "nice" - maybe powers of 2 and some others
    # Or just search 1..80 or so
    
    # For efficiency, let's try: start with {1,2,4,...,2^m} and try to insert more
    # between them or after them
    
    # Actually, let's just do a greedy construction:
    # Start with empty set, try to add elements from 1 upward
    # For each new element, check if the property still holds
    
    elements = []
    sum_by_card = [set([0])]  # only cardinality 0
    
    # Try adding elements in increasing order
    for candidate in range(1, 257):
        if time.time() > deadline:
            break
        new_sbc = _try_add(elements, candidate, sum_by_card, len(elements))
        if new_sbc is not None:
            elements.append(candidate)
            sum_by_card = new_sbc
    
    # Now try to improve by swapping
    # Local search: try replacing one element with another
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(elements)):
            if time.time() > deadline:
                break
            old = elements[i]
            for new in range(1, 257):
                if new == old or new in elements:
                    continue
                test = elements[:i] + [new] + elements[i+1:]
                test.sort()
                if _check_property(test):
                    if len(test) > len(elements) or (len(test) == len(elements) and test > elements):
                        elements = test
                        improved = True
                        break
    
    return [int(x) for x in elements]

# EVOLVE-BLOCK-END
