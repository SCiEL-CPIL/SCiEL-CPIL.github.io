import time
import random
import itertools

L = 232792560

def is_dissociated(subset):
    if not subset:
        return True
    values = [L // k for k in subset]
    sums = {0}
    for v in values:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        sums.update(new_sums)
        if len(sums) < (1 << len(subset)):
            return False
    return len(sums) == (1 << len(subset))

def can_add(subset, k):
    """
    Check if adding k to subset keeps it dissociated.
    Returns True if yes, False otherwise.
    """
    if k in subset:
        return False
    v = L // k
    # Get current sums
    values = [L // x for x in subset]
    sums = {0}
    for val in values:
        new_sums = set()
        for s in sums:
            new_sums.add(s + val)
        sums.update(new_sums)
    
    # Check if v is in sums - sums
    # i.e., exists s1, s2 in sums such that s1 - s2 == v
    # s1 == s2 + v
    for s in sums:
        if s + v in sums:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Elements to consider
    elements = list(range(1, 23))
    
    # Initial greedy attempts with different orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        [1, 2, 4, 8, 16] + [k for k in range(1, 23) if k not in [1, 2, 4, 8, 16]],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for k in order:
            if time.time() > deadline:
                break
            if can_add(current, k):
                current.append(k)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    # Random restarts
    # Try to find larger sets
    # We can also try local search: if we have a set of size M, 
    # try to swap an element to allow adding another?
    # Or just random greedy is usually good enough for this size.
    
    for _ in range(500):
        if time.time() > deadline:
            break
        random_order = elements[:]
        random.shuffle(random_order)
        current = []
        for k in random_order:
            if time.time() > deadline:
                break
            if can_add(current, k):
                current.append(k)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    # If we found a large set, try to improve it with local search
    # Move: remove one, add one (swap) to see if we can then add another?
    # This is complex. Let's stick to greedy + random restarts.
    
    return best_set
