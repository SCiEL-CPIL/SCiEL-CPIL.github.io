import time
import random

def is_valid_fast(elements):
    """
    Checks if the set of elements is valid.
    1. All non-empty subset sums are distinct.
    2. No non-empty subset sum divides another.
    """
    n = len(elements)
    if n == 0:
        return True
    
    # Generate all subset sums
    # Using a list is faster than a set for generation
    sums = [0]
    for e in elements:
        # Optimization: if we already have duplicates, we can't just check at the end
        # but generating all is fine for n<=15
        sums += [s + e for s in sums]
        
    # Filter out 0
    # We need to check for duplicates among non-zero sums
    # and divisibility
    
    # To check duplicates efficiently, we can use a set
    # But we also need the sums for divisibility check
    # Let's use a list and then convert to set?
    # Or just use a set for membership and a list for iteration?
    
    non_zero_sums = [s for s in sums if s != 0]
    
    # Check duplicates
    if len(non_zero_sums) != len(set(non_zero_sums)):
        return False
        
    sums_set = set(non_zero_sums)
    max_sum = max(sums_set)
    
    # Check divisibility
    # We iterate through unique sums
    # Optimization: skip a if 2*a > max_sum
    for a in sums_set:
        if a * 2 > max_sum:
            continue
        # Check multiples
        # Using a set lookup is O(1)
        for m in range(2 * a, max_sum + 1, a):
            if m in sums_set:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_set = []
    best_size = 0
    
    # Define search spaces, biased towards larger numbers
    # Larger numbers -> larger sums -> fewer divisibility conflicts
    search_spaces = [
        list(range(150, 257)),
        list(range(100, 257)),
        list(range(50, 257)),
        list(range(1, 257))
    ]
    
    # Try to find the largest possible set
    # We try sizes from 15 down to 1
    # For each size, we try different search spaces
    
    for target_size in range(15, 0, -1):
        if target_size <= best_size:
            break
            
        found = False
        
        for space in search_spaces:
            if time.time() > deadline:
                break
                
            # Try multiple random orderings of the space
            # We don't need to try all, just enough to find one
            attempts = 20
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                    
                # Create a shuffled copy of the search space
                candidates = space[:]
                random.shuffle(candidates)
                
                current_set = []
                
                for x in candidates:
                    if len(current_set) == target_size:
                        break
                    
                    # Check if adding x is valid
                    # Optimization: check size first? No, is_valid handles it.
                    # But we can skip if x is already in current_set (shuffled unique list ensures this)
                    
                    if is_valid_fast(current_set + [x]):
                        current_set.append(x)
                        
                if len(current_set) == target_size:
                    # Found a valid set of target_size
                    best_set = current_set
                    best_size = target_size
                    found = True
                    break
            
            if found:
                break
                
        if found:
            break # We found the max size for this space, or we can stop if we want? 
            # Actually, we are iterating target_size downwards. 
            # If we found size 15, we break the outer loop.
            # If we found size 14, we break the outer loop? 
            # Yes, because we want the MAX size. 
            # So if we find a valid set of size K, that's the best we can do for this run 
            # (assuming we tried larger K and failed).
            # Wait, the logic is: 
            # for target_size in 15..1:
            #   try to find set of size target_size
            #   if found: return it
            # So if we find 15, we return. If we fail 15, we try 14. If we find 14, we return.
            # This is correct.
            pass

    # Fallback if nothing found (should not happen, size 1 is always valid)
    if best_size == 0:
        best_set = [1]
        
    # Ensure types are correct
    return [int(x) for x in best_set]
