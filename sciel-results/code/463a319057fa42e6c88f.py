"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    N = 256
    # Precompute compatibility masks
    # mask[i] is a bitmask where bit j is 1 if i and j are compatible (distinct)
    # We only care about distinct pairs, so mask[i] should have bit i as 0.
    masks = [0] * (N + 1)
    for i in range(1, N + 1):
        m = 0
        for j in range(1, N + 1):
            if i == j:
                continue
            if (i * j) % (i + j) != 0:
                m |= (1 << j)
        masks[i] = m

    # Initial greedy construction
    # Start with all odds? Or just greedy from 1 to 256?
    # Odds are a very strong core.
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    
    best_mask = 0
    best_size = 0
    
    # Multiple greedy attempts
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Strategy: Start with odds, then add compatible evens
        current_mask = 0
        for x in odds:
            current_mask |= (1 << x)
            
        evens = [x for x in range(2, N + 1) if x % 2 == 0]
        random.shuffle(evens)
        
        for e in evens:
            if (masks[e] & current_mask) == 0:
                current_mask |= (1 << e)
        
        # Local Search / Simulated Annealing style
        # We want to maximize popcount(current_mask)
        # Try to add elements, if stuck, do swaps to unstick.
        
        local_improvement = True
        while local_improvement and time.time() < deadline:
            local_improvement = False
            
            # 1. Try to add any element
            # Iterate through all 1..256, check if compatible with current_mask
            added = False
            for x in range(1, N + 1):
                if (current_mask & (1 << x)) == 0: # x not in set
                    if (masks[x] & current_mask) == 0: # x compatible with all in set
                        current_mask |= (1 << x)
                        added = True
                        local_improvement = True
                        break # Restart search from beginning as set changed
            
            if not added:
                # 2. Try to swap to escape local optimum
                # Pick a random element in set to remove, and a random element not in set to add
                # Ensure the new element is compatible with the rest of the set
                in_set = []
                not_in_set = []
                for x in range(1, N + 1):
                    if current_mask & (1 << x):
                        in_set.append(x)
                    else:
                        not_in_set.append(x)
                
                if in_set and not_in_set:
                    # Random swap
                    for _ in range(100): # Try a few swaps
                        if time.time() > deadline: break
                        a = random.choice(in_set)
                        b = random.choice(not_in_set)
                        # Check if b is compatible with (Set - {a})
                        new_mask = current_mask & ~(1 << a)
                        if (masks[b] & new_mask) == 0:
                            # Valid swap. Accept it to explore.
                            # We don't increase size, but we might enable future additions.
                            current_mask = new_mask | (1 << b)
                            local_improvement = True
                            break
                            # Note: We don't break the outer while immediately, 
                            # we just break the swap attempt loop and go back to try adding.
        
        if bin(current_mask).count('1') > best_size:
            best_size = bin(current_mask).count('1')
            best_mask = current_mask

    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result
# EVOLVE-BLOCK-END
