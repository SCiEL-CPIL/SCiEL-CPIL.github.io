import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

# Precompute LCM of 1..22 and reciprocal values
_LCM = reduce(_lcm, range(1, 23))
_RECIPS = [0] + [_LCM // i for i in range(1, 23)]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill

    def time_left():
        return deadline - time.time()

    def is_dissociated(subset):
        """Check if all 2^k subset sums are distinct."""
        if not subset:
            return True
        sums = {0}
        for x in subset:
            r = _RECIPS[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s + r)
            sums = sums | new_sums
        return len(sums) == (1 << len(subset))

    def greedy_build(order):
        """Greedily build a valid set by iterating through order."""
        current_set = []
        sums = {0}
        for elem in order:
            if time_left() < 1.0:
                break
            r = _RECIPS[elem]
            # Check collision: is {s + r for s in sums} disjoint from sums?
            collision = False
            for s in sums:
                if (s + r) in sums:
                    collision = True
                    break
            if not collision:
                current_set.append(elem)
                new_sums = set()
                for s in sums:
                    new_sums.add(s + r)
                sums = sums | new_sums
        return current_set

    def backtrack_search(current_set, remaining, current_sums, current_size):
        """
        Backtracking search to maximize cardinality.
        current_set: list of chosen elements
        remaining: sorted list of remaining candidates
        current_sums: set of current subset sums
        current_size: current cardinality
        Returns the best set found.
        """
        if not remaining or time_left() < 0.5:
            return current_set[:]
        
        best = current_set[:]
        
        # Try to include each element in remaining
        for i, elem in enumerate(remaining):
            r = _RECIPS[elem]
            # Check collision
            collision = False
            for s in current_sums:
                if (s + r) in current_sums:
                    collision = True
                    break
            
            if not collision:
                new_sums = current_sums | {s + r for s in current_sums}
                new_set = current_set + [elem]
                new_remaining = remaining[:i] + remaining[i+1:]
                
                # Recurse
                candidate = backtrack_search(new_set, new_remaining, new_sums, current_size + 1)
                if len(candidate) > len(best):
                    best = candidate
            
            if time_left() < 0.5:
                break
        
        return best

    # Strategy 1: Greedy with multiple orderings
    best_set = []
    best_size = 0

    # Try ascending order
    cand = greedy_build(list(range(1, 23)))
    if len(cand) > best_size:
        best_size = len(cand)
        best_set = cand

    # Try descending order
    cand = greedy_build(list(range(22, 0, -1)))
    if len(cand) > best_size:
        best_size = len(cand)
        best_set = cand

    # Try random shuffles
    random.seed(42)
    for _ in range(100):
        if time_left() < 5.0:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        cand = greedy_build(order)
        if len(cand) > best_size:
            best_size = len(cand)
            best_set = cand

    # Strategy 2: Backtracking search from the best greedy result
    if best_size < 15 and time_left() > 10.0:
        # Use backtracking to try to improve
        # Start from empty set and search
        remaining = list(range(1, 23))
        # Sort by some heuristic - maybe by value descending
        remaining.sort(key=lambda x: _RECIPS[x], reverse=True)
        
        result = backtrack_search([], remaining, {0}, 0)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    # If we still have time, try more random restarts with backtracking
    if time_left() > 15.0:
        for _ in range(20):
            if time_left() < 5.0:
                break
            random.shuffle(remaining)
            result = backtrack_search([], remaining, {0}, 0)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result

    # Final verification
    if not best_set:
        best_set = [1]
    elif not is_dissociated(best_set):
        # Fallback to a known valid set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return list(best_set)
