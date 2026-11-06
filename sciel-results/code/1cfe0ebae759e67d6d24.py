"""
Strategy: Exact integer arithmetic with LCM normalization + multi-strategy search.

Key insight: LCM(1..22) = 232792560. Each 1/i maps to an integer LCM//i.
Subset sums become integer sums of these values. Distinctness is exact.

We use:
1. A fast greedy construction with multiple orderings (ascending, descending,
   random shuffles) to quickly find large valid sets.
2. A local search / hill-climbing to try to improve the cardinality by
   swapping elements or adding new ones.
3. A bounded backtracking search for the final push to maximize size.

The LCM approach makes all arithmetic exact and fast (integers up to ~2e8,
subset sums up to ~2e9, well within Python int range).
"""

import time
import random
import sys
from math import gcd

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _compute_lcm():
    """Compute LCM of 1..22."""
    result = 1
    for i in range(1, 23):
        result = result * i // gcd(result, i)
    return result


_LCM = _compute_lcm()  # 232792560
# Precompute the integer representation of 1/i
_RECIPS = [0] + [_LCM // i for i in range(1, 23)]  # _RECIPS[i] = LCM//i


def _all_sums(subset):
    """
    Given a subset (list of elements from 1..22), compute all subset sums
    as a set of integers. Returns (sums_set, count) where count should be
    2^len(subset) if all distinct.
    """
    sums = {0}
    for x in subset:
        r = _RECIPS[x]
        new_sums = set()
        for s in sums:
            new_sums.add(s + r)
        sums = sums | new_sums
    return sums


def _is_valid(subset):
    """Check if all 2^k subset sums are distinct."""
    if not subset:
        return True
    sums = _all_sums(subset)
    return len(sums) == (1 << len(subset))


def _greedy_build(order, target_k=None):
    """
    Greedily build a valid set by iterating through `order` and adding an
    element if it doesn't create a collision. If target_k is given, stop
    once we reach that size.
    """
    current_set = []
    sums = {0}
    for elem in order:
        if target_k is not None and len(current_set) >= target_k:
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
            # Update sums
            new_sums = set()
            for s in sums:
                new_sums.add(s + r)
            sums = sums | new_sums
    return current_set


def _local_search_improve(subset, max_iters=200):
    """
    Try to improve the set by:
    1. Adding any element from 1..22 not in the set.
    2. Swapping an element in the set with one not in the set (if it allows
       adding another element afterwards).
    """
    if not subset:
        return subset
    
    current = list(subset)
    current_set = set(current)
    sums = _all_sums(current)
    
    for _ in range(max_iters):
        if _time_left() < 1.0:
            break
        
        # Try adding each available element
        added = False
        for elem in range(1, 23):
            if elem in current_set:
                continue
            r = _RECIPS[elem]
            collision = False
            for s in sums:
                if (s + r) in sums:
                    collision = True
                    break
            if not collision:
                current.append(elem)
                current_set.add(elem)
                new_sums = set()
                for s in sums:
                    new_sums.add(s + r)
                sums = sums | new_sums
                added = True
                break
        
        if not added:
            # Try swaps: remove one, try to add one or more
            improved = False
            for i, elem_in in enumerate(current):
                # Try removing elem_in and seeing if we can add something better
                # Actually, just try: remove elem_in, then greedily add
                test_set = current[:i] + current[i+1:]
                test_set_set = set(test_set)
                
                # Try adding each available element to the reduced set
                for elem_add in range(1, 23):
                    if elem_add in test_set_set:
                        continue
                    # Check if test_set + [elem_add] is valid
                    test_full = test_set + [elem_add]
                    if _is_valid(test_full):
                        # This swap keeps same size but might allow future additions
                        # Only do it if it's "different" in a useful way
                        # For now, skip pure swaps that don't increase size
                        pass
            # If no addition possible, try a more aggressive local search
            # Remove the element that causes the most "tightness" and rebuild
            # Simple heuristic: try removing each element and greedily rebuilding
            best_candidate = None
            best_size = len(current)
            
            for i in range(len(current)):
                test_set = current[:i] + current[i+1:]
                # Greedily rebuild from test_set
                rebuilt = _greedy_build_from(test_set)
                if len(rebuilt) > best_size:
                    best_size = len(rebuilt)
                    best_candidate = rebuilt
            
            if best_candidate is not None:
                current = best_candidate
                current_set = set(current)
                sums = _all_sums(current)
            else:
                break
    
    return current


def _greedy_build_from(initial_subset):
    """Greedily extend an initial subset."""
    current = list(initial_subset)
    current_set = set(current)
    sums = _all_sums(current)
    
    for elem in range(1, 23):
        if elem in current_set:
            continue
        r = _RECIPS[elem]
        collision = False
        for s in sums:
            if (s + r) in sums:
                collision = True
                break
        if not collision:
            current.append(elem)
            current_set.add(elem)
            new_sums = set()
            for s in sums:
                new_sums.add(s + r)
            sums = sums | new_sums
    
    return current


def _backtrack_find(candidates, current_set, sums, target_k):
    """
    Backtracking search to find a valid set of size target_k from candidates.
    candidates: list of elements to consider (in order)
    current_set: elements chosen so far
    sums: current set of subset sums (as integers)
    """
    if _time_left() < 0.5:
        return None
    
    if len(current_set) == target_k:
        return list(current_set)
    
    remaining_needed = target_k - len(current_set)
    if len(candidates) < remaining_needed:
        return None
    
    for i, elem in enumerate(candidates):
        r = _RECIPS[elem]
        
        # Check collision
        collision = False
        for s in sums:
            if (s + r) in sums:
                collision = True
                break
        
        if collision:
            continue
        
        # Compute new sums
        new_sums = set()
        for s in sums:
            new_sums.add(s + r)
        new_sums |= sums
        
        # Recurse
        result = _backtrack_find(candidates[i+1:], current_set + [elem], new_sums, target_k)
        if result is not None:
            return result
    
    return None


def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    """
    best = [1]  # Fallback
    
    # Phase 1: Multi-ordering greedy to find a good starting set
    best_greedy = []
    
    # Try ascending
    s = _greedy_build(list(range(1, 23)))
    if len(s) > len(best_greedy):
        best_greedy = s
    
    # Try descending
    s = _greedy_build(list(range(22, 0, -1)))
    if len(s) > len(best_greedy):
        best_greedy = s
    
    # Try several random orderings
    random.seed(42)
    for _ in range(50):
        order = list(range(1, 23))
        random.shuffle(order)
        s = _greedy_build(order)
        if len(s) > len(best_greedy):
            best_greedy = s
        if _time_left() < 10.0:
            break
    
    if len(best_greedy) > len(best):
        best = list(best_greedy)
    
    # Phase 2: Local search to improve
    improved = _local_search_improve(best, max_iters=100)
    if len(improved) > len(best):
        best = improved
    
    # Phase 3: Try to push for larger sizes with backtracking
    # Start from current best size and try to go higher
    current_size = len(best)
    
    # Try sizes from 12 down to current_size + 1
    for k in range(12, current_size, -1):
        if _time_left() < 5.0:
            break
        
        # Use backtracking with multiple orderings
        found = None
        
        # Ascending
        result = _backtrack_find(list(range(1, 23)), [], {0}, k)
        if result is not None:
            found = result
        elif _time_left() > 3.0:
            # Descending
            result = _backtrack_find(list(range(22, 0, -1)), [], {0}, k)
            if result is not None:
                found = result
        elif _time_left() > 2.0:
            # Random
            order = list(range(1, 23))
            random.shuffle(order)
            result = _backtrack_find(order, [], {0}, k)
            if result is not None:
                found = result
        
        if found is not None:
            if _is_valid(found):
                best = found
                break  # Found the largest k, we're done
    
    # Final verification
    if not _is_valid(best):
        # Fallback to a known valid small set
        best = [1]
    
    return best


if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    sums = {0}
    for x in s:
        r = _RECIPS[x]
        new_sums = {ss + r for ss in sums}
        if new_sums & sums:
            print("INVALID!")
        sums = sums | new_sums
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")
    print(f"Time: {time.time() - _START_TIME:.2f}s")
