"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    N = 1000
    max_size = 0
    best_set = []
    
    # We will run multiple iterations with different random orderings
    # to find a larger set.
    # Estimated time per iteration:
    # Greedy: O(N * |S|) ~ 1000 * 90 = 90k ops.
    # Local Search: Several passes. Each pass O(N * |S|^2) ~ 8M ops.
    # 10 passes ~ 80M ops. In Python, this might take ~1-2 seconds per iteration.
    # So we can afford ~50-100 iterations in 115s.
    
    rng = random.Random(42) # Fixed seed for reproducibility in evaluation if needed, 
                            # but random is better for search. We'll use time-based or fixed?
                            # The task says "Maximize the size". Deterministic is safer for eval consistency?
                            # Usually, evaluators run once. Random is fine.
    
    for iteration in range(200):
        if time.time() > deadline - 2:
            break
            
        # 1. Greedy Phase
        # Random permutation of 1..N
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        sum_counts = np.zeros(2 * N + 1, dtype=np.int32)
        chosen = []
        chosen_set = set()
        
        for x in candidates:
            # Check if x can be added
            # Sums involved: 2x, and x+y for y in chosen
            valid = True
            if sum_counts[2 * x] >= 2:
                valid = False
            else:
                # Optimization: if chosen is large, this loop is the bottleneck.
                # We can break early.
                for y in chosen:
                    if sum_counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                sum_counts[2 * x] += 1
                for y in chosen:
                    sum_counts[x + y] += 1
                chosen.append(x)
                chosen_set.add(x)
        
        current_size = len(chosen)
        
        # 2. Local Search Phase
        # We try to improve the set size.
        # Strategy:
        # A. Try to add any missing element.
        # B. If none can be added, try swaps (remove one, add one) that might enable further additions.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # A. Try to add any element
            # To avoid O(N^2) in the worst case if we scan 1..N every time,
            # we can just scan 1..N. N=1000 is small.
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                
                valid = True
                if sum_counts[2 * x] >= 2:
                    valid = False
                else:
                    for y in chosen:
                        if sum_counts[x + y] >= 2:
                            valid = False
                            break
                
                if valid:
                    # Add x
                    sum_counts[2 * x] += 1
                    for y in chosen:
                        sum_counts[x + y] += 1
                    chosen.append(x)
                    chosen_set.add(x)
                    improved = True
                    break # Restart loop to check again
            
            if improved:
                continue
            
            # B. Try swaps
            # We look for x in chosen, y not in chosen such that:
            # 1. Removing x makes the set valid (it always is).
            # 2. Adding y to (chosen \ {x}) makes it valid.
            # 3. After swap, we might be able to add more elements.
            
            # To be efficient, we don't want to try all N*|S| swaps if they don't help.
            # But we don't know if they help until we try.
            # Let's just try a limited number of swaps per pass.
            
            # Shuffle chosen to randomize swap order
            chosen_copy = chosen[:]
            rng.shuffle(chosen_copy)
            
            swaps_tried = 0
            for x in chosen_copy:
                if time.time() > deadline - 5:
                    break
                if swaps_tried > 50: # Limit swaps per pass to keep time bounded
                    break
                    
                # Remove x
                sum_counts[2 * x] -= 1
                for y in chosen:
                    if y != x:
                        sum_counts[x + y] -= 1
                chosen.remove(x)
                chosen_set.remove(x)
                
                # Try to add a new element y
                # We prioritize y that is "small" or "random"?
                # Let's try to add any y that fits.
                # To avoid O(N) scan for each swap, we can just pick a random y?
                # No, we want to find a y that fits.
                # Let's scan a random subset of candidates or all?
                # Scanning all 1000 is 1000 * 90 = 90k ops.
                # 50 swaps * 90k = 4.5M ops. Acceptable.
                
                added_y = None
                for y in range(1, N + 1):
                    if y in chosen_set:
                        continue
                    
                    valid = True
                    if sum_counts[2 * y] >= 2:
                        valid = False
                    else:
                        for z in chosen:
                            if sum_counts[y + z] >= 2:
                                valid = False
                                break
                    
                    if valid:
                        # Add y
                        sum_counts[2 * y] += 1
                        for z in chosen:
                            sum_counts[y + z] += 1
                        chosen.append(y)
                        chosen_set.add(y)
                        added_y = y
                        break
                
                if added_y is not None:
                    # Swap was successful (size same, but different composition)
                    # Now, try to add MORE elements to see if this swap unlocks size increase
                    further_improved = False
                    for z in range(1, N + 1):
                        if z in chosen_set:
                            continue
                        valid = True
                        if sum_counts[2 * z] >= 2:
                            valid = False
                        else:
                            for w in chosen:
                                if sum_counts[z + w] >= 2:
                                    valid = False
                                    break
                        if valid:
                            sum_counts[2 * z] += 1
                            for w in chosen:
                                sum_counts[z + w] += 1
                            chosen.append(z)
                            chosen_set.add(z)
                            further_improved = True
                            break
                    
                    if further_improved:
                        # Keep the swap and the addition
                        improved = True
                        # Break out of swap loop to restart main loop
                        break
                    else:
                        # Swap didn't lead to immediate size increase.
                        # Revert the swap?
                        # If we revert, we wasted time.
                        # If we keep it, we might be stuck in a local minimum of size.
                        # However, a different composition might allow future additions.
                        # Given time constraints, let's REVERT if no immediate gain,
                        # to avoid drifting into worse local minima without progress.
                        # Remove y
                        sum_counts[2 * added_y] -= 1
                        for w in chosen:
                            if w != added_y:
                                sum_counts[added_y + w] -= 1
                        chosen.remove(added_y)
                        chosen_set.remove(added_y)
                        
                        # Re-add x
                        sum_counts[2 * x] += 1
                        for w in chosen:
                            sum_counts[x + w] += 1
                        chosen.append(x)
                        chosen_set.add(x)
                        
                else:
                    # No y found to swap with. Revert x removal.
                    sum_counts[2 * x] += 1
                    for w in chosen:
                        sum_counts[x + w] += 1
                    chosen.append(x)
                    chosen_set.add(x)
                
                swaps_tried += 1
            
            # End of swap pass
        
        # End of local search
        
        if len(chosen) > max_size:
            max_size = len(chosen)
            best_set = chosen[:]
            # print(f"Iter {iteration}: Size {max_size}") # Uncomment for debug

    return best_set
# EVOLVE-BLOCK-END
