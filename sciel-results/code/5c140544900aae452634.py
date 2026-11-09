"""Rewritten program: greedy construction with exact cardinality-indexed subset-sum DP."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

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


def _backtrack_search(elements, used, max_value, max_size, deadline, best_size, best_elements):
    """
    Backtracking search with forward checking to find larger valid sets.
    """
    current_size = len(elements)
    if current_size > best_size:
        best_elements[:] = elements[:]
        best_size[0] = current_size
    
    if current_size >= max_size:
        return best_size[0], best_elements
    
    # Try each unused candidate in ascending order
    for candidate in range(1, max_value + 1):
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        
        # Try adding this candidate
        test_elements = elements + [candidate]
        test_used = used | {candidate}
        
        if _is_cardinality_separated_fast(test_elements):
            # Recurse
            old_size = best_size[0]
            old_best = best_elements[:]
            result_size, result_best = _backtrack_search(
                test_elements, test_used, max_value, max_size, deadline, best_size, best_elements
            )
            
            # If we found something better, keep it
            # Otherwise, continue searching
            continue
    
    return best_size[0], best_elements


def _greedy_with_backtrack(max_value=256, max_size=60, deadline=None):
    """
    Hybrid approach: greedy construction with occasional backtracking to escape dead ends.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    elements = []
    used = set()
    best_size = [0]
    best_elements = []
    
    # First, do a greedy pass to get a good starting point
    for candidate in range(1, max_value + 1):
        if len(elements) >= max_size:
            break
        if time.time() > deadline:
            break
        
        if candidate in used:
            continue
        
        test_elements = elements + [candidate]
        if _is_cardinality_separated_fast(test_elements):
            elements.append(candidate)
            used.add(candidate)
    
    best_size[0] = len(elements)
    best_elements[:] = elements[:]
    
    # Now, try to improve by backtracking
    # Strategy: try removing some elements and replacing them with others
    # We'll do this by trying different subsets of the current solution
    
    # Try various pruning strategies
    prune_attempts = [
        # Remove last few elements and try different endings
        lambda els: els[:-1],
        lambda els: els[:-2],
        lambda els: els[:-3],
        # Remove middle elements
        lambda els: els[:len(els)//2] + els[len(els)//2+1:],
        lambda els: els[:len(els)//3] + els[len(els)//3+1:],
    ]
    
    for prune_fn in prune_attempts:
        if time.time() > deadline:
            break
        
        pruned = prune_fn(elements)
        pruned_used = set(pruned)
        
        # Try to extend the pruned set
        for candidate in range(1, max_value + 1):
            if len(pruned) >= max_size:
                break
            if time.time() > deadline:
                break
            
            if candidate in pruned_used:
                continue
            
            test_elements = pruned + [candidate]
            if _is_cardinality_separated_fast(test_elements):
                pruned.append(candidate)
                pruned_used.add(candidate)
        
        if len(pruned) > best_size[0]:
            best_size[0] = len(pruned)
            best_elements[:] = pruned[:]
    
    return best_elements


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


def _local_search_improve(elements, max_value, max_size, deadline):
    """
    Local search: try swapping elements to find better solutions.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    current = list(elements)
    current_size = len(current)
    best = current[:]
    best_size = current_size
    
    # Try single swaps
    for i in range(len(current)):
        if time.time() > deadline:
            break
        
        for new_val in range(1, max_value + 1):
            if time.time() > deadline:
                break
            
            if new_val in current and current[i] != new_val:
                continue
            
            # Try swapping current[i] with new_val
            test = current[:]
            test[i] = new_val
            test.sort()
            
            # Check if all elements are distinct and in range
            if len(set(test)) != len(test):
                continue
            if any(x < 1 or x > max_value for x in test):
                continue
            
            if _is_cardinality_separated_fast(test):
                # Try to extend
                extended = test[:]
                extended_used = set(extended)
                
                for candidate in range(1, max_value + 1):
                    if len(extended) >= max_size:
                        break
                    if time.time() > deadline:
                        break
                    
                    if candidate in extended_used:
                        continue
                    
                    test_ext = extended + [candidate]
                    if _is_cardinality_separated_fast(test_ext):
                        extended.append(candidate)
                        extended_used.add(candidate)
                
                if len(extended) > best_size:
                    best_size = len(extended)
                    best = extended[:]
    
    return best


def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    """
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    best = []
    
    # Strategy 1: Greedy with backtracking
    result1 = _greedy_with_backtrack(256, 60, deadline)
    if len(result1) > len(best):
        best = result1
    
    # Strategy 2: Try some heuristic starts
    heuristic_starts = [
        list(range(1, 10)),
        list(range(1, 15)),
        [1, 2, 4, 8, 16, 32, 64, 128],  # powers of 2
        list(range(2, 20)),
        list(range(10, 40)),
        list(range(50, 80)),
        list(range(1, 30)),
        list(range(5, 35)),
    ]
    
    for start in heuristic_starts:
        if time.time() > deadline:
            break
        result = _try_heuristic_start(start, 256, 60, deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Local search improvement on best so far
    if len(best) > 0 and time.time() < deadline - 10:
        improved = _local_search_improve(best, 256, 60, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
