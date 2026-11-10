import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Total number of differences to place: n_rows * (k*(k+1)//2)
    # We need to find n_rows rows, each with k+1 elements starting at 0
    # such that all positive differences are distinct.
    
    best_solution = None
    best_score = float('inf')
    
    # Helper to check if a partial row can be extended
    # For speed, we'll use a set of used differences
    
    def try_construct_random(seed):
        nonlocal best_solution, best_score, deadline
        random.seed(seed)
        
        used_diffs = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            current_max = 0
            
            # Greedy with randomization for next element
            # We want to pick the smallest possible next element that doesn't conflict
            # But sometimes picking a slightly larger one might allow better packing later?
            # Actually, for difference sets, smaller is generally better for minimization.
            # So we stick to greedy but maybe randomize the *order* of rows or initial choices?
            # The key insight from inspiration is that deterministic greedy gets stuck in local minima.
            # Randomizing the choice among *valid* next steps (if we allow skipping) or
            # randomizing the row order might help.
            
            # However, the standard greedy is: pick smallest x > last such that {x-a for a in row} are unused.
            # If we strictly follow "smallest", there's no random choice.
            # To introduce randomness, we can occasionally pick a valid x that is not the smallest,
            # or we can randomize the order in which we construct rows.
            
            # Let's try: Randomize the order of rows, and for each row, use greedy.
            # Also, if greedy gets stuck (which it shouldn't if we go high enough), we restart.
            
            # Actually, a better randomization:
            # 1. Randomize row order.
            # 2. For each row, instead of always taking the smallest valid x, 
            #    sometimes take a valid x from a small window of the smallest valid ones.
            
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            # We need to construct rows in a specific order, but the set of rows is unordered.
            # So we can just construct n_rows rows, and the order doesn't matter for the final set.
            # But the "used" set accumulates. So the order of construction matters.
            
            # Let's just construct rows one by one.
            # To add randomness, we can shuffle the "priority" of which difference to avoid? No, that's complex.
            
            # Simpler: Randomized restarts with different random seeds affecting the "greedy" choice.
            # If we always pick the smallest, it's deterministic.
            # So we must allow non-smallest choices.
            
            # Strategy: For each new element x, find the smallest valid x.
            # With probability p, accept it. With probability (1-p), look for the next valid x, etc.
            # This explores different branches.
            
            # Let's implement a randomized greedy where we pick from the first few valid options.
            
            current_row = [0]
            last_val = 0
            
            while len(current_row) < k + 1:
                # Find the next valid x
                # Start from last_val + 1
                x = last_val + 1
                valid_options = []
                found_any = False
                
                # We don't want to scan too far if it's getting expensive
                # But for small k, it's fine.
                while True:
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        valid_options.append(x)
                        found_any = True
                        # Collect a few options for randomization
                        if len(valid_options) >= 5:
                            break
                    x += 1
                    # Safety: if x gets too large, break to avoid infinite loop in this step
                    # But we expect to find one soon.
                    if x > 5000: # Arbitrary large bound for this step
                        break
                
                if not found_any:
                    return None # Failed to find any valid x, restart needed
                
                # Randomly select from the valid options found
                # Bias towards smaller options to keep the scope small
                # Weighted choice: weight = 1 / (index + 1)
                # Or just pick the smallest with high prob, random with low prob.
                
                # Let's pick the smallest with prob 0.8, random from top 5 with prob 0.2
                if random.random() < 0.8:
                    chosen_x = valid_options[0]
                else:
                    chosen_x = random.choice(valid_options)
                
                # Update state
                current_row.append(chosen_x)
                new_diffs = {chosen_x - a for a in current_row[:-1]}
                used_diffs.update(new_diffs)
                last_val = chosen_x
                
            rows.append(current_row)
            
            # Check time
            if time.time() > deadline:
                return None
        
        # All rows constructed
        # Calculate score: max entry
        score = max(max(row) for row in rows)
        
        # Verify validity (just in case, though logic should ensure it)
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return None # Invalid
                    all_diffs.add(d)
        
        return (score, rows)
    
    # Initial deterministic greedy to get a baseline quickly
    # This is the seed program logic
    def greedy_deterministic():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        score = max(max(row) for row in rows)
        return score, rows
    
    # Run deterministic first
    try:
        det_score, det_rows = greedy_deterministic()
        best_score = det_score
        best_solution = det_rows
    except Exception:
        pass
    
    # Now run randomized restarts
    num_restarts = 0
    seed = 42
    
    while time.time() < deadline:
        result = try_construct_random(seed)
        seed += 1
        num_restarts += 1
        
        if result is not None:
            score, rows = result
            if score < best_score:
                best_score = score
                best_solution = rows
                # Optional: early stop if we hit a theoretical lower bound?
                # Counting bound: sqrt(2 * total_diffs) roughly?
                # Total diffs = n_rows * k(k+1)/2
                # If best_score is very close to bound, we can stop.
                # For now, just keep searching for better.
        
        # If we've found a solution, and we're taking too long per restart, reduce randomization?
        # No, just let it run until deadline.
        
        # Adaptive: if we haven't improved in many restarts, increase randomness?
        # For now, fixed randomness.
        
    if best_solution is None:
        # Fallback: return the deterministic one if it exists, or a trivial one
        if 'det_rows' in locals() and det_rows:
            return det_rows
        # If even deterministic failed (shouldn't happen for small k), return empty or dummy
        # But task requires valid rows.
        # Let's try one more very slow greedy?
        # Or just return the deterministic result if we computed it.
        # In the code above, det_rows is local to greedy_deterministic, so we need to handle scope.
        # Let's re-run deterministic if best_solution is None.
        try:
            det_score, det_rows = greedy_deterministic()
            return det_rows
        except:
            return []
            
    return best_solution
