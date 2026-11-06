import time
import random

def construct():
    n = 512
    start_time = time.time()
    # Leave a safety margin for the evaluator's 90s limit
    time_limit = 85.0
    
    # Precompute conflicts using bitmasks for speed
    # conflict_mask[x] is a bitmask where bit y is set if x and y are in the same 4-AP
    # Actually, we need to check if adding x completes a 4-AP with existing elements.
    # Let's precompute for each x, the list of "required triples" or just use a faster check.
    
    # Faster approach:
    # For a set S (represented as a bitmask), adding x is valid if there is no 4-AP containing x where the other 3 elements are in S.
    # Let's precompute for each x, the list of masks M such that (S & M) == M means a conflict.
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # If we add x, we check all 4-APs containing x.
    # Let APs_containing[x] be a list of 3-element masks (the other 3 elements).
    # If (S & mask) == mask for any mask in APs_containing[x], then adding x is invalid.
    
    APs_containing = [None] * (n + 1)
    for x in range(1, n + 1):
        masks = []
        # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            mask = (1 << (x + d)) | (1 << (x + 2 * d)) | (1 << (x + 3 * d))
            masks.append(mask)
            
        # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            mask = (1 << (x - d)) | (1 << (x + d)) | (1 << (x + 2 * d))
            masks.append(mask)
            
        # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            mask = (1 << (x - 2 * d)) | (1 << (x - d)) | (1 << (x + d))
            masks.append(mask)
            
        # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            mask = (1 << (x - 3 * d)) | (1 << (x - 2 * d)) | (1 << (x - d))
            masks.append(mask)
            
        APs_containing[x] = masks

    def can_add(S_mask, x):
        """Check if x can be added to the set represented by S_mask."""
        for mask in APs_containing[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, order):
        """Greedily add elements from order to S_mask."""
        for x in order:
            if not (S_mask >> (x-1)) & 1: # if x not in S
                if can_add(S_mask, x):
                    S_mask |= (1 << (x-1))
        return S_mask

    def mask_to_set(mask):
        return [i for i in range(1, n+1) if (mask >> (i-1)) & 1]

    best_mask = 0
    best_size = 0
    
    random.seed(42)
    
    # Phase 1: Multiple Greedy Starts
    # Try different orders to find a good initial solution
    # 1. Standard
    m = greedy_fill(0, range(1, n+1))
    if m.bit_count() > best_size:
        best_mask = m
        best_size = m.bit_count()
        
    # 2. Reverse
    m = greedy_fill(0, range(n, 0, -1))
    if m.bit_count() > best_size:
        best_mask = m
        best_size = m.bit_count()
        
    # 3. Random Orders
    for _ in range(50):
        if time.time() - start_time > time_limit * 0.2:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        m = greedy_fill(0, order)
        if m.bit_count() > best_size:
            best_mask = m
            best_size = m.bit_count()
            
    # Phase 2: Local Search (Hill Climbing with Kicks)
    current_mask = best_mask
    
    while time.time() - start_time < time_limit:
        added_any = False
        
        # Try to add any single element
        # To speed up, we can check all elements, but 512 is small.
        # We can randomize the order of checks to find a valid one faster if stuck.
        # However, checking all 512 is fast enough with bitmasks.
        for x in range(1, n+1):
            if not (current_mask >> (x-1)) & 1:
                if can_add(current_mask, x):
                    current_mask |= (1 << (x-1))
                    added_any = True
                    if current_mask.bit_count() > best_size:
                        best_mask = current_mask
                        best_size = current_mask.bit_count()
        
        if not added_any:
            # Stuck. Perform a kick.
            # Remove 1 to 3 random elements
            k = random.randint(1, 3)
            if current_mask.bit_count() > k:
                # Get list of elements in current set
                elems = mask_to_set(current_mask)
                to_remove = random.sample(elems, k)
                for x in to_remove:
                    current_mask &= ~(1 << (x-1))
                
                # Try to fill again greedily
                # We can try a random order or standard order
                # Random order often helps escape local minima
                order = list(range(1, n+1))
                random.shuffle(order)
                current_mask = greedy_fill(current_mask, order)
                
                if current_mask.bit_count() > best_size:
                    best_mask = current_mask
                    best_size = current_mask.bit_count()
            else:
                # If set is too small to kick, restart with new random greedy
                order = list(range(1, n+1))
                random.shuffle(order)
                current_mask = greedy_fill(0, order)
                if current_mask.bit_count() > best_size:
                    best_mask = current_mask
                    best_size = current_mask.bit_count()

    return mask_to_set(best_mask)
