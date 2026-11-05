import time
import random

N = 256

def _is_valid_addition(x, selected, used_sums):
    """
    Check if x can be added to 'selected' without violating Sidon property.
    'used_sums' contains all sums a+b from 'selected' with a<=b.
    """
    if 2 * x in used_sums:
        return False
    for y in selected:
        if (x + y) in used_sums:
            return False
    return True

def _generate_maximal_sidon_set(order):
    """
    Generate a maximal Sidon set by greedily adding elements from 'order'
    if they don't conflict.
    """
    selected = []
    used_sums = set()
    
    for x in order:
        if _is_valid_addition(x, selected, used_sums):
            selected.append(x)
            # Add new sums to used_sums
            used_sums.add(2 * x)
            # x is the new largest element in selected? Not necessarily, but it's the one being added.
            # Sums are x + y for y in previous selected elements.
            for y in selected[:-1]:
                used_sums.add(x + y)
                
    return selected

def construct():
    start_time = time.time()
    # Use 55 seconds out of 60 to be safe
    deadline = start_time + 55.0
    
    # Initial deterministic solution
    order = list(range(1, N + 1))
    best_set = _generate_maximal_sidon_set(order)
    best_size = len(best_set)
    
    # Randomized search
    # We use a local random instance or seed based on time to vary runs if called multiple times?
    # The prompt implies a single call. We just run many iterations.
    
    # Optimization: Pre-compute nothing really, just fast loops.
    
    while time.time() < deadline:
        # Random permutation
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        s = _generate_maximal_sidon_set(order)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
            # Optional: If we find a size <= 16, it's very likely optimal or near-optimal.
            # But we keep searching to be sure, or until time runs out.
            
    return best_set
