"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import itertools

def _is_safe(candidate, others):
    """Return True if 1/candidate is NOT equal to sum of reciprocals of any non-empty
    subset of `others`."""
    target = Fraction(1, candidate)
    n = len(others)
    if n == 0:
        return True
    
    # Meet-in-the-middle for subset sum
    mid = n // 2
    left = others[:mid]
    right = others[mid:]
    
    # Generate all subset sums for left
    left_sums = {Fraction(0, 1)}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + Fraction(1, x)
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                # This sum equals target using only left elements
                return False
        left_sums.update(new_sums)
    
    # Check if any left sum + right sum == target
    # For each sum in right, check if (target - right_sum) is in left_sums
    right_sums = {Fraction(0, 1)}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + Fraction(1, x)
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                return False
        right_sums.update(new_sums)
    
    # Now check combinations
    for rs in right_sums:
        if rs > target:
            continue
        remainder = target - rs
        if remainder in left_sums:
            # We need to ensure the combined subset is non-empty.
            # Since target > 0, and all terms are positive, if remainder == 0 and rs == target, 
            # that's a valid subset (non-empty since target > 0).
            # The empty set sum is 0. If rs=0 and remainder=target, it implies target in left_sums.
            # If rs=target and remainder=0, it implies target in right_sums.
            # In all cases where target > 0, a match implies a non-empty subset.
            return False
            
    return True

def _is_valid_full(subset):
    """Check if the entire subset is reciprocal-subsum-free."""
    s = set(subset)
    for x in s:
        others = list(s - {x})
        if not _is_safe(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Strategy 1: Greedy construction from 24 down to 1
    # This works because adding in decreasing order ensures that for any existing element x,
    # the new element k < x, so 1/k > 1/x. Thus 1/x - 1/k < 0, and it's impossible to form
    # 1/x using k and any subset of the existing elements (since all sums are positive).
    # So we only need to check the new element k against the existing set.
    
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_safe(i, list(selected)):
            selected.add(i)
    
    best_set = set(selected)
    
    # Strategy 2: Try to add back some excluded elements
    excluded = set(range(1, 25)) - selected
    for x in sorted(excluded, reverse=True):
        if time.time() > deadline:
            break
        new_candidate = selected | {x}
        if _is_valid_full(new_candidate):
            selected = new_candidate
            best_set = set(selected)
    
    # Strategy 3: Local search - try swapping elements
    # If we have time, try to improve by removing one element and adding another
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_list = sorted(selected)
            excluded = set(range(1, 25)) - selected
            
            for x in current_list:
                if time.time() > deadline - 5:
                    break
                for y in sorted(excluded, reverse=True):
                    if y in selected:
                        continue
                    new_set = selected - {x} | {y}
                    if _is_valid_full(new_set):
                        # This swap maintains the same size but might allow future additions
                        # For now, we just track it as a potential improvement if it enables more
                        # Actually, we want to maximize cardinality, so same-size swaps don't help directly
                        # unless they enable adding more elements later. Let's skip for now.
                        pass
    
    # Strategy 4: More aggressive search - try different orderings
    # Try greedy from different starting points or with different criteria
    # For now, let's try a few variations
    
    # Variation: Start with {13..24} and try to add smaller numbers
    if time.time() < deadline - 20:
        base_set = set(range(13, 25))
        if _is_valid_full(base_set):
            for i in range(12, 0, -1):
                if time.time() > deadline - 10:
                    break
                new_set = base_set | {i}
                if _is_valid_full(new_set):
                    base_set = new_set
            if len(base_set) > len(best_set):
                best_set = base_set
    
    # Variation: Try to build from scratch with a different heuristic
    # e.g., prioritize elements that are "harder" to match
    
    # Final check: ensure non-empty
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
