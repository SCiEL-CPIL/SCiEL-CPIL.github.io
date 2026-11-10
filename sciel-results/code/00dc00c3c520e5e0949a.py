"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct."""
        k = len(elements)
        if k == 0:
            return True
        reciprocals = [Fraction(1, x) for x in elements]
        # Generate all subset sums incrementally
        sums = {Fraction(0, 1)}
        for r in reciprocals:
            new_sums = set()
            for s in sums:
                ns = s + r
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return len(sums) == (1 << k)
    
    def try_order(order, current_best):
        """Try to build a dissociated set following the given order."""
        current = []
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            test_set = current + [elem]
            if check_dissociated(test_set):
                current = test_set
                if len(current) > len(current_best):
                    current_best = current[:]
        return current_best
    
    candidates = list(range(1, 23))
    best_set = []
    
    # Strategy 1: Descending order (larger denominators first - smaller reciprocals)
    order1 = sorted(candidates, reverse=True)
    best_set = try_order(order1, best_set)
    
    # Strategy 2: Ascending order
    order2 = sorted(candidates)
    best_set = try_order(order2, best_set)
    
    # Strategy 3: Spaced out (alternating small and large)
    order3 = []
    for i in range(11):
        order3.append(i + 1)
        order3.append(22 - i)
    best_set = try_order(order3, best_set)
    
    # Strategy 4: Powers of 2 pattern (known to be dissociated)
    order4 = [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17, 15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3]
    best_set = try_order(order4, best_set)
    
    # Strategy 5: Randomized attempts
    rng = random.Random(42)
    for attempt in range(5):
        if time.time() > deadline:
            break
        random_order = candidates[:]
        rng.shuffle(random_order)
        best_set = try_order(random_order, best_set)
    
    # Strategy 6: Start with known good set, try to add more
    if len(best_set) < 12:
        base = [1, 2, 4, 8, 16]
        remaining = [x for x in candidates if x not in base]
        # Try adding remaining elements in descending order
        current = base[:]
        for elem in sorted(remaining, reverse=True):
            if time.time() > deadline:
                break
            test_set = current + [elem]
            if check_dissociated(test_set):
                current = test_set
        if len(current) > len(best_set):
            best_set = current[:]
    
    if not best_set:
        # Fallback to known good set
        best_set = [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17]
    
    return best_set
# EVOLVE-BLOCK-END
