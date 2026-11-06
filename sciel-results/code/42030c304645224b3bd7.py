"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add_fast(selected, x):
    """Fast check if x can be added to selected without violating constraints.
    
    Key insight: When checking if we can add x to selected S:
    1. For each a in S: need to check no triple from (S\{a}) ∪ {x} sums to multiple of a
       - Triples entirely in S\{a}: already valid in S
       - Triples with x and two from S\{a}: check (s1+s2+x) % a != 0 for all pairs
       - Triple with x and one from S\{a} and... wait, need 3 distinct others
    2. For x itself: need no triple from S sums to multiple of x
    
    So we only need to check:
    - For each a in S: for each pair (s1,s2) in S\{a}: (s1+s2+x) % a != 0
    - For x: for each triple in S: sum(triple) % x != 0
    """
    n = len(selected)
    selected_set = set(selected)
    
    # Check x against existing triples
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            return False
    
    # Check each existing element a against pairs + x
    for i in range(n):
        a = selected[i]
        # For each pair from S\{a}, check if (pair_sum + x) % a == 0
        for j in range(n):
            if j == i:
                continue
            s1 = selected[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                s2 = selected[k]
                if (s1 + s2 + x) % a == 0:
                    return False
    
    return True

def _greedy(order, deadline):
    """Greedy construction with given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Local search: try to add more elements or swap."""
    selected = list(selected)
    
    # Try to add elements - sort by some heuristic (larger first might be easier)
    candidates = [x for x in range(1, 193) if x not in set(selected)]
    candidates.sort(reverse=True)  # Try larger elements first
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x not in set(selected):
                if _can_add_fast(selected, x):
                    selected.append(x)
                    candidates.remove(x)
                    improved = True
    
    # Try swaps: remove one element, try to add two new ones
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        removed = selected[i]
        remaining = selected[:i] + selected[i+1:]
        
        # Try to add two elements to remaining
        candidates_for_swap = [x for x in range(1, 193) if x not in set(remaining) and x != removed]
        candidates_for_swap.sort(reverse=True)
        
        added = []
        for x in candidates_for_swap:
            if time.time() > deadline:
                break
            if len(added) == 2:
                break
            if _can_add_fast(remaining + added, x):
                added.append(x)
        
        if len(added) >= 2:
            new_selected = remaining + added
            if len(new_selected) > len(selected):
                selected = new_selected
    
    return selected

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts."""
    best = []
    start_time = time.time()
    
    # Try several random orderings
    for attempt in range(20):
        if time.time() > deadline:
            break
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        result = _greedy(rand_order, deadline)
        if len(result) > len(best):
            best = result
    
    # Also try structured orderings
    structured_orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        # Alternating high/low
        list(range(96, 0, -1)) + list(range(97, 193)),
        # Primes-like: numbers with few divisors
        [x for x in range(192, 0, -1) if x > 64],
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    # Phase 1: Randomized greedy to find good initial solution
    best = _random_greedy(deadline)
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: More local improvement rounds
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy(list(range(192, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END
