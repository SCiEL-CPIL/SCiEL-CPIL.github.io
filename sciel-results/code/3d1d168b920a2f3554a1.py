import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # 1. Greedy Initialization (Largest First)
    # This matches the seed logic, ensuring we start at least as good as the seed.
    selected = []
    reachable = {0}
    
    for x in range(100, 0, -1):
        # Check if adding x creates a sum of 200
        # 200 is created if 200-x is in reachable
        if (200 - x) not in reachable:
            selected.append(x)
            # Update reachable sums
            # New sums are s + x for s in reachable
            # We only care about sums <= 200
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        # If (200-x) is in reachable, we skip x
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # Helper to check validity and get reachable sums
    def get_reachable(s):
        r = {0}
        for x in s:
            new_r = {v + x for v in r if v + x <= 200}
            r |= new_r
            if 200 in r:
                return r, False
        return r, True

    # 2. Local Search
    # Try to add excluded elements by removing conflicting ones
    excluded = [x for x in range(1, 101) if x not in best_set]
    
    # Shuffle for diversity
    random.shuffle(excluded)
    
    while time.time() < deadline:
        # Try to add a random excluded element
        # We can't just pick one, we need to try to improve size
        # If we add x and it's valid, size increases.
        # If invalid, we might swap.
        
        # Randomly pick an element from excluded
        if not excluded:
            break
            
        idx = random.randint(0, len(excluded) - 1)
        x = excluded[idx]
        
        # Check if x can be added directly
        current_reachable, is_valid = get_reachable(best_set)
        if (200 - x) not in current_reachable:
            # Valid addition
            best_set.add(x)
            excluded.remove(x)
            best_size += 1
            continue
        
        # If not valid directly, try to find a conflict to resolve
        # We need to remove some elements from best_set to make x valid.
        # Specifically, we need to remove elements that contribute to the sum 200-x.
        # This is a hitting set problem on the subsets of best_set that sum to 200-x.
        
        # Heuristic: Try to remove one element 'a' from best_set such that
        # adding x and removing a results in a valid set.
        # This is a swap. Size stays same.
        # To increase size, we would need to remove 0 (handled above) or 
        # remove 1 and add 2 (complex).
        
        # Let's try swaps to reach a better local optimum, 
        # hoping that a future addition becomes possible.
        
        # Actually, a simpler improvement:
        # Try all pairs (x in excluded, a in best_set) ? Too slow (100*50=5000 checks, each O(N*S) validation).
        # 5000 * 20000 = 100M ops. Might be tight but doable in C, slow in Python.
        
        # Optimized validation:
        # We can maintain the reachable set incrementally?
        # No, removal is hard.
        
        # Let's just try random swaps.
        # Pick random a from best_set.
        if best_set:
            a = random.choice(list(best_set))
            # Try swap: remove a, add x
            new_set = set(best_set)
            new_set.remove(a)
            new_set.add(x)
            
            # Validate new_set
            # Fast check:
            r, valid = get_reachable(new_set)
            if valid:
                # Swap is valid. Size is same.
                # Does this help? It might open up opportunities for future additions.
                best_set = new_set
                # Update excluded list
                excluded.remove(x)
                excluded.append(a)
                # Shuffle excluded to maintain randomness
                random.shuffle(excluded)
    
    return list(best_set)
