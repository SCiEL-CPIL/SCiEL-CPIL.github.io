"""Program using a model-and-optimize approach for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_all_sums(selected):
    """
    Compute the set of all unordered triple sums for a given set of integers.
    Returns None if a collision is found, otherwise returns the set of sums.
    """
    n = len(selected)
    if n == 0:
        return set()
    
    sums = set()
    # i <= j <= k
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = selected[i] + selected[j] + selected[k]
                if s in sums:
                    return None
                sums.add(s)
    return sums

def _check_validity(selected):
    """Check if the set has unique triple sums."""
    return _compute_all_sums(selected) is not None

def _construct_base3_greedy(limit, max_size=80):
    """
    Construct a set using a base-3 inspired greedy strategy.
    Start with powers of 3 (which are guaranteed B_3[1]) and then try to
    fill in gaps with smaller numbers to increase density.
    """
    # Base 3 powers up to limit
    base_elements = []
    val = 1
    while val <= limit:
        base_elements.append(val)
        val *= 3
    
    # The powers of 3 are a valid B_3[1] set.
    # We want to replace some large powers with multiple smaller numbers to increase cardinality.
    # Or, we can start with a subset of powers and greedily add numbers.
    
    # Let's try a greedy approach starting from an empty set, but biased towards
    # numbers that are "compatible" with the base-3 structure?
    # Actually, a better structural change is to use the base-3 set as a seed
    # and then perform local search to swap elements for better packing.
    
    # Let's just return the base-3 set for now, or a greedy extension of it.
    # The number of base-3 elements up to 4096 is small (3^0..3^7 = 1,3,9,27,81,243,729,2187 -> 8 elements).
    # We need up to 80. So we need a denser strategy.
    
    # Strategy:
    # 1. Start with a random set of size K (e.g., 10) that is valid.
    # 2. Greedily add elements.
    # 3. If stuck, try to swap an element with a smaller one to free up sum space.
    
    # Let's implement a greedy that prioritizes smaller numbers to maximize count.
    selected = []
    used_sums = set()
    
    # Sort candidates by value
    candidates = list(range(1, limit + 1))
    
    # Try to add the smallest available number that doesn't cause a collision
    for x in candidates:
        if len(selected) >= max_size:
            break
        if _can_add(selected, x, used_sums):
            selected.append(x)
            _add_sums(selected, x, used_sums)
            
    return selected

def _can_add(selected, x, used_sums):
    """Check if x can be added to selected without collision."""
    n = len(selected)
    new_sums = set()
    
    # (x, x, x)
    s = 3 * x
    if s in used_sums or s in new_sums:
        return False
    new_sums.add(s)
    
    # (a, x, x)
    for a in selected:
        s = a + 2 * x
        if s in used_sums or s in new_sums:
            return False
        new_sums.add(s)
        
    # (a, b, x)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j] + x
            if s in used_sums or s in new_sums:
                return False
            new_sums.add(s)
            
    return True

def _add_sums(selected, x, used_sums):
    """Add all new sums involving x to used_sums."""
    n = len(selected)
    used_sums.add(3 * x)
    for a in selected:
        used_sums.add(a + 2 * x)
    for i in range(n):
        for j in range(i, n):
            used_sums.add(selected[i] + selected[j] + x)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    limit = 4096
    max_size = 80
    
    best_result = []
    
    # Run multiple random greedy attempts with different biases
    for seed in range(50):
        random.seed(seed)
        
        # Strategy: Randomized greedy with preference for smaller numbers
        # but with some randomness to explore different structures.
        candidates = list(range(1, limit + 1))
        random.shuffle(candidates)
        
        # Sort by a heuristic: prefer smaller numbers, but with noise
        # score = x + random_noise
        scored_candidates = [(x + random.uniform(0, 100), x) for x in candidates]
        scored_candidates.sort(key=lambda t: t[0])
        ordered_candidates = [x for _, x in scored_candidates]
        
        selected = []
        used_sums = set()
        
        for x in ordered_candidates:
            if len(selected) >= max_size:
                break
            if time.time() > deadline - 5:
                break
            if _can_add(selected, x, used_sums):
                selected.append(x)
                _add_sums(selected, x, used_sums)
                
        if len(selected) > len(best_result):
            best_result = selected
            if len(best_result) >= max_size:
                break
                
    return best_result

# EVOLVE-BLOCK-END
