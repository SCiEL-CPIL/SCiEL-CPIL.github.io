"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    import random
    
    # Deadline guard: 90s wall-clock limit, keep safety margin
    deadline = time.time() + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Initial high-quality seed: Greedy fill
    selected = []
    total = Fraction(0, 1)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    
    # Ensure non-empty (greedy always picks at least 1/d for d>=2)
    if not selected:
        selected = [0]
        total = reciprocals[0]
        
    best_indices = selected[:]
    best_deficit = 1 - total
    best_total = total
    
    # Local search: Hill climbing
    # Moves: 
    # 1. Add an unselected index if sum < 1
    # 2. Remove a selected index if sum > 0 (to allow swaps or just to reduce? No, we want to minimize deficit, so we want sum close to 1 from below. Removing increases deficit, so only useful if it enables a better add/swap)
    # 3. Swap a selected index with an unselected one if sum remains < 1
    
    selected_set = set(selected)
    
    # Precompute list of indices
    all_indices = list(range(n))
    
    # Optimization: Keep track of current sum as Fraction
    # To speed up, we can use floats for quick rejection, but must verify with Fraction.
    
    # Heuristic: Sort candidates by value? Or just random order?
    # Let's use a time-bounded loop.
    
    # To make it faster, we can pre-compute float values
    float_recips = [float(r) for r in reciprocals]
    current_float_sum = sum(float_recips[i] for i in selected)
    
    while time.time() < deadline:
        improved = False
        
        # Shuffle for diversity in local search
        random.shuffle(all_indices)
        
        # Try all possible moves
        # 1. Add moves
        for i in all_indices:
            if i in selected_set:
                continue
            # Quick float check
            if current_float_sum + float_recips[i] >= 1.0:
                continue
            
            # Exact check
            new_total = total + reciprocals[i]
            if new_total < 1:
                new_deficit = 1 - new_total
                if new_deficit < best_deficit:
                    # Accept
                    best_deficit = new_deficit
                    best_total = new_total
                    best_indices = selected + [i]
                    
                    # Update current state for next iterations in this pass?
                    # Usually local search updates state immediately or at end of pass.
                    # Let's update immediately for steepest descent.
                    selected = best_indices
                    selected_set.add(i)
                    total = new_total
                    current_float_sum = float(new_total)
                    improved = True
                    # Break out of add loop? Or continue?
                    # Steepest descent: pick best move. 
                    # But iterating all adds to find the BEST add is expensive.
                    # Let's just accept the first improvement to allow more moves per second.
                    # Or better: find the best add, then apply it.
                    # Given 47 items, checking all adds is 47 ops. Fast.
                    # Let's find the best add in this pass.
                    # Actually, the loop above breaks on first improvement if we structure it that way.
                    # Let's restructure: Find best add, best remove (if useful?), best swap.
                    # Since we are maximizing sum (minimizing deficit), we only add if sum < 1.
                    # Removing only helps if we can then add something bigger? 
                    # Swap is equivalent to remove + add.
                    
                    # Let's restart the inner loop logic to find the BEST move among all candidates.
                    # This is more robust.
                    pass
        
        # The above logic was getting messy. Let's implement a cleaner "Find Best Move" strategy.
        
        # Re-evaluate current state
        # current state: selected, total, current_float_sum
        
        best_move_deficit = best_deficit
        best_move_indices = None
        best_move_total = None
        
        # 1. Find Best Add
        for i in all_indices:
            if i in selected_set:
                continue
            if current_float_sum + float_recips[i] >= 1.0:
                continue
            new_total = total + reciprocals[i]
            if new_total < 1:
                d = 1 - new_total
                if d < best_move_deficit:
                    best_move_deficit = d
                    best_move_indices = selected + [i]
                    best_move_total = new_total
        
        # 2. Find Best Swap
        # For each selected j, for each unselected i
        # If total - recip[j] + recip[i] < 1 and > current total? (We want to increase sum)
        # Actually, we just want to minimize deficit.
        # So we want to maximize new_total subject to new_total < 1.
        
        # Optimization: Only swap if it increases the sum.
        # new_total = total - recip[j] + recip[i]
        # We want new_total > total (to reduce deficit)
        # So recip[i] > recip[j]
        
        # Pre-sort unselected by reciprocal value descending to break early?
        # Unselected indices:
        unselected = [i for i in all_indices if i not in selected_set]
        
        # Sort unselected by value descending (largest first) to find good swaps quickly?
        # Or just iterate. 47*47 is small.
        
        for j in selected:
            # If we remove j, we gain "space" of recip[j]
            # We can add any i such that recip[i] <= recip[j] + (1 - total) ?
            # Condition: total - recip[j] + recip[i] < 1
            # recip[i] < 1 - total + recip[j]
            # Let max_allowed_i_val = 1 - total + recip[j]
            
            # We want to maximize recip[i] subject to recip[i] < max_allowed_i_val and i not in selected
            # And we want the resulting sum to be greater than current total.
            # total - recip[j] + recip[i] > total  =>  recip[i] > recip[j]
            
            # So we look for i in unselected such that:
            # recip[j] < recip[i] < 1 - total + recip[j]
            
            # If such i exists, the best one is the one with largest recip[i] satisfying the upper bound.
            
            # Let's iterate unselected to find best candidate for this j
            # Since n is small, simple iteration is fine.
            
            # Optimization: Sort unselected by value desc once outside?
            # Let's just do a simple loop.
            
            # We can break early if we find a swap that results in deficit 0? (Impossible, strict < 1)
            
            for i in unselected:
                # Check if i > j in value
                if reciprocals[i] <= reciprocals[j]:
                    continue
                # Check if sum < 1
                # total - recip[j] + recip[i] < 1
                # This is equivalent to recip[i] < 1 - total + recip[j]
                
                # Exact check
                new_total = total - reciprocals[j] + reciprocals[i]
                if new_total < 1:
                    d = 1 - new_total
                    if d < best_move_deficit:
                        best_move_deficit = d
                        # New indices: replace j with i
                        best_move_indices = [x for x in selected if x != j] + [i]
                        best_move_total = new_total
        
        # Apply best move if found
        if best_move_indices is not None:
            selected = best_move_indices
            selected_set = set(selected)
            total = best_move_total
            current_float_sum = float(total)
            best_deficit = best_move_deficit
            best_indices = selected[:]
            
            # If deficit is very small, we might stop early?
            # But we have time.
            
        else:
            # No improvement found in this pass.
            # To avoid local minima, we could do a random restart or random walk.
            # Given the structure, greedy + local search is quite strong.
            # Let's try a random perturbation: pick a random selected element and remove it,
            # then try to add the best possible element.
            
            # This allows escaping local minima where no single add/swap helps, but a remove+add sequence does.
            
            if len(selected) > 1:
                # Remove a random element
                j_to_remove = random.choice(selected)
                
                # Tentative state
                temp_selected = [x for x in selected if x != j_to_remove]
                temp_set = set(temp_selected)
                temp_total = total - reciprocals[j_to_remove]
                
                # Now try to add the best possible element to temp_selected
                # We want to maximize sum < 1.
                # This is the "Greedy Fill" on the remaining set.
                # Actually, just find the best single add to temp_selected that keeps it < 1.
                # Or better, perform a mini greedy fill from temp_selected?
                # No, just find the best single add first. If that doesn't improve, maybe try 2 adds?
                # Let's stick to single add for simplicity and speed.
                
                best_add_deficit = 1 - temp_total # Current deficit of temp state (larger than before)
                best_add_indices = None
                best_add_total = temp_total
                
                # Find best add i not in temp_set
                for i in all_indices:
                    if i in temp_set:
                        continue
                    if float(temp_total) + float_recips[i] >= 1.0:
                        continue
                    new_total = temp_total + reciprocals[i]
                    if new_total < 1:
                        d = 1 - new_total
                        if d < best_add_deficit:
                            best_add_deficit = d
                            best_add_indices = temp_selected + [i]
                            best_add_total = new_total
                            
                # If this perturbation leads to a better state than current best, accept
                if best_add_indices is not None and best_add_deficit < best_deficit:
                    selected = best_add_indices
                    selected_set = set(selected)
                    total = best_add_total
                    current_float_sum = float(total)
                    best_deficit = best_add_deficit
                    best_indices = selected[:]
                    
        # Check time again
        if time.time() > deadline - 2.0:
            break
            
    return best_indices
# EVOLVE-BLOCK-END
