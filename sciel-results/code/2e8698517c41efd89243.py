import time
from itertools import combinations

def _validate_set(elements):
    """Check if a set of elements satisfies the divisibility-incomparable subset sum condition."""
    n = len(elements)
    if n == 0:
        return True
    # Generate all non-empty subset sums
    subset_sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        subset_sums.add(s)
    # Check no two distinct sums divide each other
    sums_list = sorted(subset_sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return False
    return True

def _fast_validate(elements, current_sums):
    """Check if adding a new element to a set with known subset sums is valid."""
    n = len(elements)
    new_elements = elements + [new_elem]
    # Generate all subset sums for the new set
    new_sums = set()
    for mask in range(1, 1 << (n + 1)):
        s = 0
        for i in range(n + 1):
            if mask & (1 << i):
                s += new_elements[i]
        new_sums.add(s)
    # Check divisibility
    sums_list = sorted(new_sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return False
    return True

def _greedy_with_validation(candidates, target_size, deadline):
    """Greedy approach: try to add elements one by one, validating each time."""
    current_set = []
    for elem in candidates:
        if time.time() > deadline:
            break
        if len(current_set) >= target_size:
            break
        # Try adding this element
        test_set = current_set + [elem]
        if _validate_set(test_set):
            current_set = test_set
    return current_set

def _local_search_improve(initial_set, all_candidates, deadline, target_size):
    """Try to improve a set by swapping elements."""
    current = list(initial_set)
    current_size = len(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one element and adding another
        for remove_idx in range(len(current)):
            if time.time() > deadline:
                break
            for add_elem in all_candidates:
                if add_elem in current:
                    continue
                test_set = current[:remove_idx] + current[remove_idx+1:] + [add_elem]
                test_set.sort()
                if len(set(test_set)) != len(test_set):
                    continue
                if _validate_set(test_set):
                    # Check if this improves (larger size or different composition)
                    if len(test_set) > current_size:
                        current = test_set
                        current_size = len(current)
                        improved = True
                        break
                    # Even same size, try different sets for future improvements
                    elif _validate_set(test_set):
                        current = test_set
                        improved = True
                        break
            if improved:
                break
    return current

def construct():
    deadline = time.time() + 50  # 50 second deadline, 10s safety margin
    
    # Strategy: Use elements in a range where subset sums are large and close
    # Key insight: if all elements are > 100, subset sums are > 100, 
    # and we can avoid small divisibility relations
    
    best_set = []
    best_size = 0
    
    # Approach 1: Greedy from large values (to keep sums large)
    candidates_desc = list(range(256, 0, -1))
    result = _greedy_with_validation(candidates_desc, 30, deadline)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result
    
    # Approach 2: Use elements in [128, 256] - sums are large, less likely to have divisibility
    if time.time() < deadline and best_size < 30:
        large_candidates = list(range(256, 127, -1))
        result = _greedy_with_validation(large_candidates, 30, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Approach 3: Local search to improve
    if best_size < 30 and time.time() < deadline:
        all_candidates = list(range(1, 257))
        improved = _local_search_improve(best_set, all_candidates, deadline, 30)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Ensure we return at least something valid
    if not best_set:
        # Fallback: try a known small valid set
        for test_set in [[2, 3], [3, 5], [5, 7, 11]]:
            if _validate_set(test_set):
                best_set = test_set
                break
    
    # Convert to list of Python ints
    return [int(x) for x in sorted(best_set)]
