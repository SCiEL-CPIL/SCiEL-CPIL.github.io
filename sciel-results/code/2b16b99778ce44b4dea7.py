"""
Rewritten program: Iterated Local Search with fast incremental validity checks.
Combines the fast divisibility check from inspiration 1 with a more robust
search strategy (ILS) to escape local optima found by simple greedy methods.
"""
import random
import time

# EVOLVE-BLOCK-START

def _get_subset_sums(elements):
    """
    Returns a set of all non-empty subset sums of the given elements.
    """
    if not elements:
        return set()
    sums = {0}
    for x in elements:
        # Add x to all existing sums
        new_sums = {s + x for s in sums}
        sums |= new_sums
    sums.discard(0)
    return sums

def _is_valid_set(elements):
    """
    Checks if the set of elements satisfies:
    1. All non-empty subset sums are unique.
    2. No two distinct subset sums divide one another.
    """
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
            
    sums = _get_subset_sums(elements)
    max_sum = max(sums) if sums else 0
    
    # Check divisibility
    # For each sum s, check if any multiple k*s (k>=2) is in sums
    for s in sums:
        if s == 0:
            continue
        # Only need to check up to max_sum
        for k in range(2, max_sum // s + 1):
            if s * k in sums:
                return False
    return True

def _check_addition(elements, x, current_sums):
    """
    Checks if adding x to elements (which have current_sums) keeps validity.
    Returns the new set of sums if valid, None otherwise.
    """
    if x in elements:
        return None
        
    # Generate new sums
    # New sums = current_sums U {s + x for s in current_sums} U {x}
    new_sums = set(current_sums)
    new_sums.add(x)
    
    # Check for uniqueness (collision)
    # If any s+x is already in current_sums or equals x or equals another s'+x, it's invalid.
    # More simply: just build the set and check size.
    # But we need to be fast.
    
    # Let's just build the candidate set and check validity on it.
    # Since the size is small, this is acceptable if the check is fast.
    
    # Optimization: 
    # 1. Check if x creates a collision.
    #    x cannot be in current_sums.
    #    s+x cannot be in current_sums for any s in current_sums.
    #    s+x cannot equal s'+x for s != s' (impossible if sums are unique).
    
    if x in current_sums:
        return None
        
    for s in current_sums:
        if s + x in current_sums:
            return None
            
    # Now we have a candidate set of sums.
    # Let's create the actual set to check divisibility.
    # This might be slightly slower than a pure check, but allows us to return the new sums.
    candidate_sums = set(current_sums)
    candidate_sums.add(x)
    for s in current_sums:
        candidate_sums.add(s + x)
        
    # Now check divisibility on candidate_sums
    max_sum = max(candidate_sums)
    for s in candidate_sums:
        if s == 0:
            continue
        for k in range(2, max_sum // s + 1):
            if s * k in candidate_sums:
                return None
                
    return candidate_sums

def _local_search(start_elements, rng, deadline):
    """
    Hill climbing local search. Tries to add elements or swap elements to increase size.
    """
    current = list(start_elements)
    current_sums = _get_subset_sums(current)
    best = list(current)
    best_size = len(current)
    
    # Pool of available numbers
    pool = [x for x in range(1, 257) if x not in current]
    
    stuck_count = 0
    t0 = time.time()
    
    while time.time() < deadline:
        # Try to add an element
        added = False
        # Shuffle pool to randomize order
        rng.shuffle(pool)
        for x in pool:
            if time.time() > deadline:
                break
            new_sums = _check_addition(current, x, current_sums)
            if new_sums is not None:
                current.append(x)
                current_sums = new_sums
                pool.remove(x)
                added = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
                break # Restart the loop to try adding another
        
        if not added:
            # Try a swap: remove one element, add another
            if not pool:
                break
            swapped = False
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                if not current:
                    break
                idx = rng.randrange(len(current))
                x_remove = current[idx]
                x_add = rng.choice(pool)
                
                # Check if removing x_remove and adding x_add is valid
                # This is complex. Easier: 
                # 1. Remove x_remove from current.
                # 2. Try to add x_add to the reduced set.
                
                temp_elements = current[:idx] + current[idx+1:]
                temp_sums = _get_subset_sums(temp_elements)
                
                new_sums = _check_addition(temp_elements, x_add, temp_sums)
                if new_sums is not None:
                    # Valid swap
                    current = temp_elements + [x_add]
                    current_sums = new_sums
                    pool.remove(x_add)
                    if x_remove not in pool:
                        pool.append(x_remove)
                    swapped = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                    break
            
            if not swapped:
                stuck_count += 1
                if stuck_count > 20:
                    break
        else:
            stuck_count = 0
            
    return best

def construct():
    deadline = time.time() + 85  # Safety margin
    rng = random.Random(42)
    
    best_result = []
    
    # Initial best from seed/known good
    # [31, 32, 34] is size 3.
    # Let's try to find something better.
    
    # Strategy: Multiple restarts of ILS
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random starting set
        # Start with a small valid set to ensure we don't start invalid
        # Try picking random numbers and checking validity
        start_set = []
        available = list(range(1, 257))
        rng.shuffle(available)
        
        # Greedy build a small valid set
        for x in available:
            if len(start_set) >= 5:
                break
            cand = start_set + [x]
            if _is_valid_set(cand):
                start_set.append(x)
                
        if not start_set:
            # Fallback to a known small valid set if greedy fails (unlikely)
            start_set = [31]
            
        # Run local search
        result = _local_search(start_set, rng, deadline)
        
        if len(result) > len(best_result):
            best_result = result
            
        # If we hit max possible or high size, break
        if len(best_result) >= 15: # Unlikely to go that high, but safe
            break
            
    # Final validation
    if _is_valid_set(best_result):
        return [int(x) for x in best_result]
    
    # Fallback
    return [31, 32, 34]

# EVOLVE-BLOCK-END
