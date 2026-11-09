"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
import numpy as np

_LCM = 232792560

def _get_weights(subset):
    return np.array([_LCM // x for x in subset], dtype=np.int64)

def _can_add(subset, x):
    """Check if adding x to subset keeps it dissociated."""
    if x in subset:
        return False
    if not subset:
        return True
        
    weights = _get_weights(subset)
    k = len(subset)
    n = 1 << k
    
    # Compute all subset sums of current subset
    sums = np.zeros(n, dtype=np.int64)
    for i in range(k):
        bit = 1 << i
        for start in range(bit, n, 2 * bit):
            sums[start:start + bit] += weights[i]
            
    # Create a set for O(1) lookup
    sum_set = set(sums.tolist())
    
    w_x = _LCM // x
    
    # Check if s + w_x is in sum_set for any s in sum_set
    # This checks if s1 = s2 + w_x has a solution
    for s in sum_set:
        if (s + w_x) in sum_set:
            return False
            
    return True

def _construct_greedy(universe, deadline):
    """Build a dissociated set by adding elements one by one."""
    current = []
    remaining = list(universe)
    
    # Shuffle to break symmetry
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        if _can_add(current, x):
            current.append(x)
            
    return current

def _local_search(start_set, deadline):
    """Local search to maximize size."""
    current = list(start_set)
    best = list(current)
    best_size = len(current)
    
    universe = set(range(1, 23))
    
    t = time.time()
    
    while time.time() < deadline:
        # Try to add an element
        not_in = [x for x in universe if x not in current]
        added = False
        for x in not_in:
            if _can_add(current, x):
                current.append(x)
                added = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
                break
        
        if added:
            continue
        
        # If no addition, try swaps
        if not current:
            break
            
        # Try a few random swaps
        swapped = False
        for _ in range(5):
            if time.time() > deadline:
                break
            i = random.randint(0, len(current) - 1)
            y = current[i]
            
            if not not_in:
                continue
            x = random.choice(not_in)
            
            # Check swap
            new_set = [v for v in current if v != y]
            new_set.append(x)
            
            # Check if new_set is dissociated
            # We can use _can_add on new_set \ {x} with x?
            # No, _can_add checks if x can be added to a set.
            # Here we removed y.
            # Let's just use a fast check.
            # Since we just swapped, the size is the same.
            # We can check if new_set is dissociated by checking if x can be added to (current \ {y}).
            # This is exactly what _can_add does!
            base_set = [v for v in current if v != y]
            if _can_add(base_set, x):
                current = new_set
                swapped = True
                not_in = [x for x in universe if x not in current]
                break
                
        if swapped:
            continue
            
        # If no swap worked, try removing an element to allow addition
        if not current:
            break
        i = random.randint(0, len(current) - 1)
        y = current[i]
        new_set = [v for v in current if v != y]
        
        # Check if this allows an addition
        added_after_remove = False
        not_in_after_remove = [x for x in universe if x not in new_set]
        for x in not_in_after_remove:
            if _can_add(new_set, x):
                new_set.append(x)
                added_after_remove = True
                break
        
        if added_after_remove:
            current = new_set
            if len(current) > best_size:
                best_size = len(current)
                best = list(current)
        else:
            # Stuck, restart or break
            break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    universe = list(range(1, 23))
    best = []
    best_size = 0
    
    # Run multiple greedy constructions with different shuffles
    for i in range(50):
        if time.time() > deadline - 10:
            break
        random.seed(i) # Reproducible shuffles
        result = _construct_greedy(universe, deadline - 10)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            
        # If we found a large set, try to improve it
        if best_size >= 16:
             # Run local search on this best set
             improved = _local_search(best, deadline - 5)
             if len(improved) > best_size:
                 best_size = len(improved)
                 best = improved

    # Final polish
    not_in = [x for x in universe if x not in best]
    best = _try_add_elements(best, not_in, deadline - 1)
    
    return best

def _try_add_elements(best, removed, deadline):
    """Try to add back removed elements one by one."""
    current = list(best)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in removed:
            if time.time() > deadline:
                break
            if x not in current:
                if _can_add(current, x):
                    current.append(x)
                    removed.remove(x)
                    improved = True
                    break
    return current
