import time

def _subset_sums_distinct(elements):
    """
    Check if all subset sums of the given elements are distinct.
    """
    sums = {0}
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def construct():
    """
    Returns a set of 12 distinct positive integers with distinct subset sums.
    Uses a greedy approach to minimize the largest element.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 12
    current_set = []
    used = set()
    
    # Greedy construction
    for i in range(n):
        if time.time() > deadline:
            # Fallback to known optimal solution if time runs out
            return [1 << i for i in range(12)]
            
        # Try smallest positive integer not in used
        candidate = 1
        while True:
            if candidate in used:
                candidate += 1
                continue
            
            # Check if adding candidate preserves distinct subset sums
            test_set = current_set + [candidate]
            if _subset_sums_distinct(test_set):
                current_set.append(candidate)
                used.add(candidate)
                break
            else:
                candidate += 1
                
            if time.time() > deadline:
                # Fallback to known optimal solution if time runs out
                return [1 << i for i in range(12)]

    return current_set
