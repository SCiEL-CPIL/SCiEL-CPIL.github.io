import time
import random

N = 512

def can_add(x, mask):
    # mask is an int where bit i represents presence of i
    # Indices 1..N
    
    # Case 1: x is the first term (a)
    # x, x+d, x+2d, x+3d
    # Max d: x + 3d <= N => d <= (N - x) // 3
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1) and ((mask >> (x + 3 * d)) & 1):
            return False
            
    # Case 2: x is the second term (a+d)
    # x-d, x, x+d, x+2d
    # Max d: x - d >= 1 => d <= x - 1
    max_d = x - 1
    for d in range(1, max_d + 1):
        if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
            return False
            
    # Case 3: x is the third term (a+2d)
    # x-2d, x-d, x, x+d
    # Max d: x - 2d >= 1 => d <= (x - 1) // 2
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
            return False
            
    # Case 4: x is the fourth term (a+3d)
    # x-3d, x-2d, x-d, x
    # Max d: x - 3d >= 1 => d <= (x - 1) // 3
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x - 3 * d)) & 1) and ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
            return False
            
    return True

def greedy_fill(mask, order):
    for x in order:
        if can_add(x, mask):
            mask |= (1 << x)
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_mask = 0
    best_size = 0
    
    # We will run multiple rounds of (Random Greedy + Local Search)
    # Each round starts with a fresh random greedy construction
    
    while time.time() < deadline - 2.0:
        # 1. Random Greedy Initialization
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask = greedy_fill(0, order)
        
        # 2. Local Search
        improved = True
        # Limit local search steps per round to avoid getting stuck in one local optimum
        # for too long, though the time check handles it.
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 2a. Try to add any valid element
            # We scan all x not in mask
            added_any = False
            for x in range(1, N + 1):
                if not (mask >> x) & 1:
                    if can_add(x, mask):
                        mask |= (1 << x)
                        added_any = True
            if added_any:
                improved = True
                continue
                
            # 2b. If stuck, try a 1-removal move
            # Get list of elements in mask
            # Optimization: We can maintain a list, but rebuilding is O(N) which is small
            elements = [x for x in range(1, N + 1) if (mask >> x) & 1]
            if not elements:
                break
                
            # Try a few random removals to escape local minima
            # Just one is standard, but trying 2-3 might be better if fast
            # Let's stick to 1 for speed, the outer loop provides diversity
            y = random.choice(elements)
            old_mask = mask
            mask &= ~(1 << y)
            
            # Refill greedily with a new random order
            refill_order = list(range(1, N + 1))
            random.shuffle(refill_order)
            mask = greedy_fill(mask, refill_order)
            
            if mask.bit_count() > old_mask.bit_count():
                improved = True
            else:
                mask = old_mask
                
        # Update global best
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask
            
    # Final polish
    for x in range(1, N + 1):
        if not (best_mask >> x) & 1:
            if can_add(x, best_mask):
                best_mask |= (1 << x)
                
    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    return result
