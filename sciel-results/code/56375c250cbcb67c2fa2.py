import itertools
import time
import random

def is_dissociated(subset):
    if not subset:
        return True
    # L = lcm(1..22)
    L = 232792560
    values = [L // k for k in subset]
    sums = set()
    # Use iterative subset sum generation
    # Start with {0}
    sums.add(0)
    for v in values:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        sums.update(new_sums)
        if len(sums) < (1 << len(subset)):
            # Early termination: if we have fewer sums than expected, there's a collision
            # Wait, the check should be: if len(sums) < 2^i where i is the number of elements processed so far
            # Actually, we just need to check if len(sums) == 2^m at the end.
            # But we can check earlier.
            # Let's just do it simply for small m.
            pass
    return len(sums) == (1 << len(subset))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin
    
    # Try to find a set of size 12
    # Start with a greedy set
    # Sort by L/k descending? Or just try random subsets
    
    best_set = []
    best_size = 0
    
    # Try all subsets of size 12? Too many.
    # Try random subsets of size 12
    # Or start with {1, 2, 4, 8, 16} and add more
    
    base = [1, 2, 4, 8, 16]
    candidates = [k for k in range(1, 23) if k not in base]
    
    # Greedy add
    current = list(base)
    for k in candidates:
        if time.time() > deadline:
            break
        new_set = current + [k]
        if is_dissociated(new_set):
            current = new_set
    
    best_set = current
    best_size = len(current)
    
    # Try to improve with random swaps/adds
    # Or just try to find any set of size 13
    
    # Let's try random search for size 12 and 13
    # Since we found a size 12 set, let's see if we can find 13.
    
    # Heuristic: elements with large k (small L/k) are easier to add.
    # So maybe prefer larger k.
    
    # Let's try a local search
    # State: a set of size m
    # Move: remove one, add one
    
    # Start with the greedy set
    current = list(best_set)
    m = len(current)
    
    # Try to add one more
    for k in range(22, 0, -1): # Try larger k first (smaller values)
        if k in current:
            continue
        new_set = current + [k]
        if is_dissociated(new_set):
            current = new_set
            m += 1
            break # Restart the loop to try adding another
    
    # If we have size 12, try to find size 13
    # If we have size 13, try to find size 14
    
    # Let's do a more systematic search for size 13
    # Pick 13 elements from 22
    # There are 1.3 million such sets.
    # Checking each takes 2^13 = 8192 steps.
    # 1.3M * 8192 is too slow.
    
    # But we can prune.
    # If a subset of size 12 is not dissociated, no superset is.
    # We can iterate over combinations of size 13, and check their subsets of size 12?
    # No, that's complex.
    
    # Let's just use the greedy result.
    # The greedy result with base {1,2,4,8,16} and adding candidates in order 3,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22.
    
    # Let's re-run the greedy with a better order: sort by L/k ascending (i.e., k descending).
    # This adds "small" numbers first, which might leave room for "large" numbers?
    # Or sort by L/k descending (k ascending).
    
    # Let's try both.
    
    def greedy_search(order):
        current = []
        for k in order:
            if time.time() > deadline:
                break
            new_set = current + [k]
            if is_dissociated(new_set):
                current = new_set
        return current
    
    order1 = list(range(1, 23))
    order2 = list(range(22, 0, -1))
    
    set1 = greedy_search(order1)
    set2 = greedy_search(order2)
    
    best_set = set1 if len(set1) > len(set2) else set2
    best_size = len(best_set)
    
    # Try random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
        random_order = list(range(1, 23))
        random.shuffle(random_order)
        s = greedy_search(random_order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    return best_set
