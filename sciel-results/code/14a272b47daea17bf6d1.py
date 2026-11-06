"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    
    Strategy:
    1. Greedy seed: iterate from 100 down to 1, add x if it doesn't create a subset sum of 200.
    2. Local search: try to swap out smaller elements for larger ones that were excluded,
       or add back excluded elements by removing others.
    3. Deadline guard: return best found before 90s wall-clock limit.
    """
    deadline = time.time() + 85.0  # safety margin
    
    # --- Phase 1: Greedy seed ---
    # We want to maximize cardinality. The greedy approach from large to small
    # tends to include large numbers which "waste" sum capacity, leaving room
    # for more small numbers. But actually, for avoiding a specific target sum,
    # we should be careful.
    
    # Let's think about this differently. We want a subset S of [1,100] such that
    # no subset of S sums to exactly 200. We want |S| as large as possible.
    
    # Key insight: if we include too many numbers, we're likely to have a subset
    # summing to 200. The total sum of [1,100] is 5050, so 200 is relatively small.
    
    # Greedy approach: start with all numbers, and remove the minimum number of
    # elements to break all subset sums of 200. This is equivalent to finding a
    # minimum hitting set for all subsets that sum to 200, which is hard.
    
    # Alternative: build up the set greedily. Start with empty set, add elements
    # in some order, maintaining the invariant that no subset sums to 200.
    
    # The order matters. If we add small numbers first, we might block many large
    # numbers. If we add large numbers first, we might have more flexibility.
    
    # Let's try: add numbers from 1 to 100, but skip any that would create a subset
    # sum of 200.
    
    # Actually, let's use a smarter approach. We'll maintain the set of reachable
    # subset sums (capped at 200). When considering adding x, we check if 200-x
    # is already reachable (which would mean adding x creates a subset sum of 200).
    
    # Greedy from 1 to 100:
    selected = set()
    reachable = {0}  # set of all subset sums achievable, capped at 200
    
    for x in range(1, 101):
        # Check if adding x would create a subset sum of 200
        # This happens if 200 - x is in reachable
        if (200 - x) not in reachable:
            selected.add(x)
            # Update reachable sums
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            reachable |= new_sums
    
    best_set = set(selected)
    best_size = len(selected)
    
    # --- Phase 2: Local search to improve ---
    # Try to swap elements: remove a small element and add a larger one that was excluded.
    # Or try different orderings.
    
    # First, let's try the reverse greedy (from 100 down to 1) for comparison
    selected2 = set()
    reachable2 = {0}
    for x in range(100, 0, -1):
        if (200 - x) not in reachable2:
            selected2.add(x)
            new_sums = set()
            for s in reachable2:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            reachable2 |= new_sums
    
    if len(selected2) > best_size:
        best_set = set(selected2)
        best_size = len(selected2)
    
    # Try local search: for each element in best_set, try removing it and adding
    # an excluded element that fits.
    excluded = set(range(1, 101)) - best_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for to_remove in list(best_set):
            # Try removing to_remove
            candidate = set(best_set)
            candidate.discard(to_remove)
            
            # Recompute reachable for candidate
            reach = {0}
            for x in sorted(candidate):
                new_sums = set()
                for s in reach:
                    ns = s + x
                    if ns <= 200:
                        new_sums.add(ns)
                reach |= new_sums
            
            # Try adding each excluded element
            for to_add in sorted(excluded, reverse=True):
                if to_add in candidate:
                    continue
                if (200 - to_add) not in reach:
                    # Adding to_add is safe
                    new_candidate = set(candidate)
                    new_candidate.add(to_add)
                    if len(new_candidate) > best_size:
                        best_set = new_candidate
                        best_size = len(best_set)
                        excluded = set(range(1, 101)) - best_set
                        improved = True
                        break
            if improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
