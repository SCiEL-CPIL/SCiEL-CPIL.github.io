"""Model-and-optimize: maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated (all reciprocal subset sums distinct) iff adding each new
reciprocal never produces a subset-sum collision. We build greedily over random
orderings of [1..22] using exact Fraction arithmetic, with a wall-clock deadline
guard, and keep the largest valid set found.
"""
import time
import random
from fractions import Fraction

_DEADLINE = 75.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _check_add(sums, r):
    """Check if adding reciprocal r to set with subset sums `sums` creates collision."""
    for s in sums:
        if s + r in sums:
            return False
    return True

def _add_element(sums, r):
    """Add element with reciprocal r to the set, updating sums in place."""
    new_sums = [s + r for s in sums]
    sums.update(new_sums)

def _try_greedy(order):
    """Greedily add elements in `order`, keeping a set only if no sum collision.
    Returns the final list (possibly small)."""
    sums = {Fraction(0)}          # current subset sums (normalized Fractions)
    chosen = []
    for a in order:
        r = Fraction(1, a)
        if _check_add(sums, r):
            _add_element(sums, r)
            chosen.append(a)
    return chosen

def _try_greedy_with_local_search(order, best_k):
    """Greedily add elements in `order`, then try local search to improve.
    Returns the final list (possibly small)."""
    sums = {Fraction(0)}          # current subset sums (normalized Fractions)
    chosen = []
    chosen_set = set()
    
    # First pass: greedy
    for a in order:
        if a in chosen_set:
            continue
        r = Fraction(1, a)
        if _check_add(sums, r):
            _add_element(sums, r)
            chosen.append(a)
            chosen_set.add(a)
    
    if len(chosen) <= best_k:
        return chosen
    
    # Local search: try swaps to improve
    full_set = set(range(1, 23))
    current_set = set(chosen)
    
    for iteration in range(50):
        if _deadline_hit():
            break
        
        # Try 1-for-1 swaps
        found_improvement = False
        for remove_a in list(current_set):
            # Compute sums without remove_a
            sums_without = {Fraction(0)}
            valid = True
            for b in sorted(current_set - {remove_a}):
                r_b = Fraction(1, b)
                if not _check_add(sums_without, r_b):
                    valid = False
                    break
                _add_element(sums_without, r_b)
            
            if not valid:
                continue
            
            # Try to add an element not in current_set
            for add_a in sorted(full_set - current_set):
                r_add = Fraction(1, add_a)
                if _check_add(sums_without, r_add):
                    # Found a valid swap
                    current_set.remove(remove_a)
                    current_set.add(add_a)
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if found_improvement:
            continue
        
        # Try 2-for-1 swaps (remove 2, add 1) to escape local optima
        found_improvement = False
        elements = list(current_set)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                remove_a, remove_b = elements[i], elements[j]
                # Compute sums without remove_a and remove_b
                sums_without = {Fraction(0)}
                valid = True
                for b in sorted(current_set - {remove_a, remove_b}):
                    r_b = Fraction(1, b)
                    if not _check_add(sums_without, r_b):
                        valid = False
                        break
                    _add_element(sums_without, r_b)
                
                if not valid:
                    continue
                
                # Try to add an element not in current_set
                for add_a in sorted(full_set - current_set):
                    r_add = Fraction(1, add_a)
                    if _check_add(sums_without, r_add):
                        # Found a valid 2-for-1 swap
                        current_set.remove(remove_a)
                        current_set.remove(remove_b)
                        current_set.add(add_a)
                        found_improvement = True
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return list(current_set)

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    # Deterministic seed baseline (known valid, 9 elements).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    # Random restarts: greedy over random permutations with local search.
    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _try_greedy_with_local_search(order, best_k)
        if len(cand) > best_k:
            best_k = len(cand)
            best = cand
            # If we reached the theoretical-ish cap, keep going a bit more,
            # but stop early if we've found a very large set to save time.
            if best_k >= 14:
                break

    if best is None or len(best) == 0:
        best = [1]
    return list(best)
