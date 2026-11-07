"""Improved program for cardinality-separated subset sums."""
import time
from itertools import combinations

P = {'n': 256, 'maximum_size': 60}

def _is_valid_cardinality_separated(elements):
    """Check if the set satisfies the cardinality-separated subset sum property.
    
    For any two subsets of different cardinalities, their sums must differ.
    """
    n = len(elements)
    # sum_sets[k] = set of all subset sums using exactly k elements
    sum_sets = [set() for _ in range(n + 1)]
    sum_sets[0].add(0)
    
    # For each element, update sum sets (but this is exponential in n)
    # Better: iterate over all subsets via bitmasks for small n
    # For n up to ~20, 2^n is manageable; for larger, we need a different approach
    
    if n <= 20:
        for mask in range(1 << n):
            k = bin(mask).count('1')
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sum_sets[k].add(s)
    else:
        # For larger n, use dynamic programming by cardinality
        # sums[k] = set of achievable sums with exactly k elements
        for i, elem in enumerate(elements):
            for k in range(i + 1, 0, -1):
                new_sums = set()
                for s in sum_sets[k - 1]:
                    new_sums.add(s + elem)
                sum_sets[k].update(new_sums)
    
    # Check: no sum appears in two different cardinality sets
    seen = {}
    for k in range(n + 1):
        for s in sum_sets[k]:
            if s in seen:
                return False
            seen[s] = k
    return True

def _try_greedy_build(deadline):
    """Greedy: add smallest available element that maintains validity."""
    best_set = []
    used = set()
    
    for candidate in range(1, 257):
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        
        test_set = best_set + [candidate]
        if _is_valid_cardinality_separated(test_set):
            best_set = test_set
            used.add(candidate)
            if len(best_set) >= P['maximum_size']:
                break
    
    return best_set

def _try_powers_of_two():
    """Powers of 2 guarantee uniqueness of subset sums, hence cardinality separation."""
    result = []
    p = 1
    while p <= 256:
        result.append(p)
        p *= 2
    return result

def _try_randomized_search(deadline):
    """Randomized search with local improvement to find larger valid sets."""
    best_set = []
    best_size = 0
    
    # Start with powers of 2 as a baseline
    baseline = _try_powers_of_two()
    if len(baseline) > best_size:
        best_set = baseline
        best_size = len(baseline)
    
    # Try greedy first
    greedy_set = _try_greedy_build(deadline)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Try random restarts with greedy completion
    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        # Start with a random subset of small numbers, then greedily add
        start_elements = set()
        # Pick a random starting point
        max_start = min(30, 256)
        for _ in range(random.randint(1, 5)):
            e = random.randint(1, max_start)
            start_elements.add(e)
        
        current = sorted(start_elements)
        
        # Verify the start is valid
        if not _is_valid_cardinality_separated(current):
            continue
        
        # Greedily add the rest
        for candidate in range(1, 257):
            if time.time() > deadline:
                break
            if candidate in set(current):
                continue
            test_set = current + [candidate]
            if _is_valid_cardinality_separated(test_set):
                current = test_set
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            if best_size >= P['maximum_size']:
                break
    
    return best_set

def construct():
    deadline = time.time() + 80  # Safety margin: kill at 90s
    
    # Try quick deterministic methods first
    result = _try_powers_of_two()
    if len(result) >= P['maximum_size']:
        return list(result)
    
    # Try greedy build
    greedy_result = _try_greedy_build(deadline)
    if len(greedy_result) > len(result):
        result = greedy_result
    
    if len(result) >= P['maximum_size']:
        return list(result)
    
    # Try randomized search with remaining time
    if time.time() < deadline - 5:
        random_result = _try_randomized_search(deadline)
        if len(random_result) > len(result):
            result = random_result
    
    # Ensure we return a valid list of Python ints
    result = [int(x) for x in result]
    return result

# EVOLVE-BLOCK-END
