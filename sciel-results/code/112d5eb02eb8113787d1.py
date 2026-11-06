import random
import time

def _greedy(order):
    chosen = []
    counts = [0] * 2001
    
    for x in order:
        valid = True
        if counts[2*x] > 1:
            valid = False
        else:
            for y in chosen:
                s = x + y
                if counts[s] > 1:
                    valid = False
                    break
        
        if valid:
            chosen.append(x)
            counts[2*x] += 1
            for y in chosen[:-1]:
                s = x + y
                counts[s] += 1
                
    return chosen, counts

def _local_search(chosen, counts):
    # chosen is a list, counts is the array from greedy
    # We want to try swaps: remove x, add y
    # To be efficient, we should track which sums are "tight" (count == 2)
    # But a simple approach: 
    # For each x in chosen:
    #   Tentatively remove x (update counts)
    #   Try to add some y not in chosen
    #   If valid, keep it
    #   Else revert
    
    chosen_set = set(chosen)
    improved = True
    while improved:
        improved = False
        # Sort chosen to have a deterministic order or random? Random is better for search.
        # We iterate through a copy of chosen to allow removals
        for i in range(len(chosen)):
            if i >= len(chosen): break
            x = chosen[i]
            
            # Tentatively remove x
            # We need to decrement counts for all sums involving x
            # Sums: 2x, and x+y for y in chosen, y != x
            # Note: In the current count array, count[s] includes all pairs.
            # Removing x reduces count[2x] by 1.
            # For each y in chosen, y != x: count[x+y] decreases by 1.
            
            # Instead of updating the global counts array and reverting (which is error prone),
            # let's just check validity of a new set S' = S \ {x} U {y} from scratch?
            # No, that's O(N^2) per swap. Too slow.
            
            # Let's use the counts array.
            # Store old counts? No, just modify and revert.
            
            # 1. Decrement counts for x
            old_counts = counts[:] # Copying 2001 ints is fast (~10us)
            
            counts[2*x] -= 1
            for y in chosen:
                if y != x:
                    counts[x+y] -= 1
            
            # 2. Try to add a new y
            # Which y to try? 
            # Ideally, try y's that were previously rejected.
            # Or just try random y's from 1..1000 not in chosen.
            # To keep it fast, let's try a limited set of candidates.
            # Candidates: numbers in 1..1000 not in chosen.
            # We can pre-compute a list of available numbers?
            # Or just iterate 1..1000.
            
            found_swap = False
            # We prefer larger y to maximize sum range? Or smaller?
            # Let's try random candidates to escape local minima.
            # Generate a few random candidates
            candidates = [y for y in range(1, 1001) if y not in chosen_set]
            random.shuffle(candidates)
            
            # Limit candidates to keep it fast? 
            # If we try all 930 candidates for each of 70 elements, that's 65k checks.
            # Each check is O(|S|) ~ 70. Total 4.5M ops. OK.
            
            for y in candidates:
                # Check if y can be added to current (reduced) set
                if counts[2*y] > 1:
                    continue
                valid = True
                for z in chosen:
                    if z == x: continue # x is removed
                    s = y + z
                    if counts[s] > 1:
                        valid = False
                        break
                if valid:
                    # Accept swap
                    # Update counts for adding y
                    counts[2*y] += 1
                    for z in chosen:
                        if z == x: continue
                        counts[y+z] += 1
                    
                    # Update chosen list
                    chosen[i] = y
                    chosen_set.remove(x)
                    chosen_set.add(y)
                    found_swap = True
                    improved = True
                    break
            
            # Revert counts if no swap found or even if found (we modified in place)
            # Wait, if found_swap, we already updated counts for the new y.
            # If not found, we must revert the removal of x.
            if not found_swap:
                # Revert removal of x
                counts[2*x] += 1
                for z in chosen:
                    if z == x: continue
                    counts[x+z] += 1
            # If found_swap, the counts are already correct for the new set.
            # But note: we used `old_counts`? No, we modified `counts` directly.
            # So if found_swap, `counts` is correct.
            # If not found_swap, we reverted, so `counts` is correct.

    return chosen, counts

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_set = []
    best_size = 0
    
    # Run baseline
    order = list(range(1, 1001))
    s, c = _greedy(order)
    best_set = s
    best_size = len(s)
    
    # Random restarts with local search
    while time.time() < deadline:
        order = list(range(1, 1001))
        random.shuffle(order)
        s, c = _greedy(order)
        
        # Apply local search
        s, c = _local_search(s, c)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    return best_set
