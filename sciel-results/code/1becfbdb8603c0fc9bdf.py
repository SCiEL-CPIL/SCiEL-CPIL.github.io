import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    timeout = 110.0 # Leave margin for evaluator overhead
    
    # 1. Greedy Initialization
    def greedy_row(used_diffs):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs.update(new_diffs)
        return row

    used = set()
    rows = []
    for _ in range(n_rows):
        r = greedy_row(used)
        rows.append(r)
    
    current_scope = max(max(r) for r in rows)
    
    # Helper to get diffs of a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # Re-optimization of a single row
    # Finds the row with the smallest max element that is valid given 'forbidden'
    def optimize_row(forbidden):
        # We want to minimize max(a)
        # Lower bound for max element:
        # Minimal Golomb ruler of length k+1.
        # For k=7 (8 marks), min Golomb ruler is 11? No, G(8)=11 is for 8 marks? 
        # G(4)=6, G(5)=11, G(6)=17, G(7)=25, G(8)=34.
        # So min max is at least 34.
        # But we have forbidden diffs, so it will be larger.
        
        best_row = None
        best_max = float('inf')
        
        # Try increasing M
        # Upper bound: current_scope (or slightly less)
        # Start M from a reasonable lower bound
        min_M = 0
        # A quick lower bound: max(forbidden) + 1? No.
        # Just start from 0 and go up? Too slow.
        # Start from the max element of the previous row if available? 
        # We don't pass the previous row here, but we can infer from context.
        # Let's start from 0 and increment. To speed up, we can jump.
        
        # Actually, we can just run backtracking without fixing M, and track the min M found.
        # Backtracking:
        # a[0] = 0
        # Choose a[1] ... a[k]
        
        def backtrack(idx, current_row, current_diffs):
            nonlocal best_row, best_max
            
            if time.time() - start_time > timeout:
                return False # Abort

            if idx == k + 1:
                if current_row[-1] < best_max:
                    best_max = current_row[-1]
                    best_row = current_row[:]
                return True # Found a better one, but keep searching for even better? 
                           # Or return True to stop? 
                           # We want the BEST, so we should continue or use branch and bound.
                           # If we find one with max M, we can prune any branch where current_max >= M.
                # But since we build left to right, current_row[-1] is the current max.
                # We can prune if current_row[-1] >= best_max.
            
            # Pruning: if current max is already >= best_max, no point.
            if current_row[-1] >= best_max:
                return False
            
            # MRV: Choose next index. Since we build sequentially, it's idx.
            # To use MRV properly, we should choose the variable with smallest domain.
            # But sequential is simpler. Let's stick to sequential for now, 
            # but add strong pruning.
            
            last_val = current_row[-1]
            # Next value must be > last_val
            # Also, we need to ensure we can fit the remaining (k - idx) elements.
            # Min step is 1. So we need space.
            
            # Iterate possible next values
            # Start from last_val + 1
            # Upper bound for next value:
            # We need to leave room for remaining elements.
            # Also, we don't want to exceed best_max.
            max_possible = best_max - 1 # If we exceed, it won't be better
            
            # Heuristic: Start from smaller values
            for x in range(last_val + 1, max_possible + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                current_row.append(x)
                current_diffs.update(new_diffs)
                
                # Recursive call
                if backtrack(idx + 1, current_row, current_diffs):
                    # If we found a solution, we could return True to stop early if we just want ANY solution,
                    # but we want the BEST. So we don't return True immediately, 
                    # we let it explore other x values that might yield a smaller max?
                    # No, x is increasing. If we found a solution with x, 
                    # larger x will result in larger or equal max.
                    # So if we found a solution, we can break?
                    # Wait, a larger x might allow a smaller *final* max? No, the row is increasing.
                    # The final max is at least x.
                    # So if we found a valid row ending with some value, 
                    # trying larger x for the current step will only increase the potential max.
                    # Thus, if we find a valid completion, it's the best for this prefix?
                    # No, a different prefix (smaller x) might lead to a smaller overall max.
                    # But we are iterating x from small to large.
                    # So the first valid x that leads to a solution is the best for this step?
                    # Yes, because any subsequent x is larger, so the final max will be >= this x.
                    # And the previous x's were invalid.
                    # So we can return True to stop exploring larger x for this idx?
                    # Yes.
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
                    return True
                
                current_row.pop()
                current_diffs.difference_update(new_diffs)
            
            return False

        # Initialize backtrack
        # We need to set an initial best_max to allow pruning.
        # We can use the max element of the row being replaced as an initial upper bound.
        # But we don't have it here. Let's pass it in.
        
        return best_row

    # Modify optimize_row to accept an upper bound
    def optimize_row_bounded(forbidden, upper_bound):
        best_row = None
        best_max = upper_bound
        
        def backtrack(idx, current_row, current_diffs):
            nonlocal best_row, best_max
            if time.time() - start_time > timeout:
                return False
            if idx == k + 1:
                if current_row[-1] < best_max:
                    best_max = current_row[-1]
                    best_row = current_row[:]
                return False # Continue searching for smaller? No, we iterate x from small to large.
                             # The first solution found is the smallest for this path.
                             # But there might be a different path with a smaller max?
                             # Since we pick smallest x first, the first complete row found
                             # is the lexicographically smallest, which tends to have small max.
                             # But is it the absolute minimum max?
                             # Not necessarily. A row with larger early elements might allow smaller later elements?
                             # No, elements are increasing.
                             # So the first valid row found (greedy) is likely the best.
                             # Backtracking is mainly to handle cases where greedy fails?
                             # Greedy always succeeds if a solution exists? No, greedy can get stuck.
                             # So backtracking explores alternatives.
            
            if current_row[-1] >= best_max:
                return False
            
            last_val = current_row[-1]
            # We want to find a row with max < best_max
            # So x must be < best_max
            for x in range(last_val + 1, best_max):
                new_diffs = {x - a for a in current_row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & current_diffs:
                    continue
                
                current_row.append(x)
                current_diffs.update(new_diffs)
                
                if backtrack(idx + 1, current_row, current_diffs):
                    # Found a better solution
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
                    return True # Stop, we found the best possible for this prefix?
                                # Actually, since we go from small to large, 
                                # any other x will be larger, so the max will be larger.
                                # So this is the best.
                
                current_row.pop()
                current_diffs.difference_update(new_diffs)
            
            return False

        backtrack(1, [0], set())
        return best_row

    # Main Loop
    no_improve_count = 0
    while time.time() - start_time < timeout:
        improved = False
        # Pick a random row to optimize
        r_idx = random.randint(0, n_rows - 1)
        old_row = rows[r_idx]
        old_diffs = get_diffs(old_row)
        
        # Remove old diffs
        used.difference_update(old_diffs)
        
        # Try to find a better row
        # Upper bound: current_scope
        new_row = optimize_row_bounded(used, current_scope)
        
        if new_row is not None:
            new_diffs = get_diffs(new_row)
            # Add new diffs
            used.update(new_diffs)
            
            # Check if scope improved
            new_scope = max(max(r) for r in rows) # Need to update rows first
            rows[r_idx] = new_row
            new_scope = max(max(r) for r in rows)
            
            if new_scope < current_scope:
                current_scope = new_scope
                improved = True
                no_improve_count = 0
            else:
                # Rollback if no improvement? 
                # Or keep it if it's the same? 
                # If it's the same, it might help future steps? 
                # But we want to minimize scope. 
                # If scope didn't drop, we might be in a local optimum.
                # Let's rollback to avoid cycling?
                # Or accept with some probability?
                # For now, rollback if no improvement to focus on reducing scope.
                used.difference_update(new_diffs)
                used.update(old_diffs)
                rows[r_idx] = old_row
                no_improve_count += 1
        else:
            # No better row found, restore old
            used.update(old_diffs)
            no_improve_count += 1
        
        # If stuck, perturb
        if no_improve_count > 50:
            # Perturb: Pick a row, increase a random element, re-greedy
            p_idx = random.randint(0, n_rows - 1)
            p_row = rows[p_idx]
            p_diffs = get_diffs(p_row)
            used.difference_update(p_diffs)
            
            # Create a perturbed version
            # Pick a random index to "push" up
            if len(p_row) > 1:
                push_idx = random.randint(1, len(p_row)-1)
                # Increase p_row[push_idx] by a random amount
                # But we need to re-greedy the whole row to maintain validity?
                # Let's just re-greedy with a "hint" or constraint.
                # Simplest: Just re-greedy. It will likely produce the same row.
                # To force change, we can forbid the previous row's differences?
                # No, that might make it impossible.
                # Let's try: Re-greedy, but skip the first candidate for one element.
                
                # Alternative Perturb: 
                # Take the row, remove it.
                # Add a random difference to 'used' temporarily? No.
                
                # Let's try: Re-greedy, but start x from a higher value for one step.
                # Or just re-greedy and hope for the best? 
                # If re-greedy gives the same row, we are stuck.
                
                # Better Perturb: 
                # Select a random element in the row, increase it by 1, 
                # then fix the rest of the row greedily from that point?
                # This is complex.
                
                # Simple Perturb: 
                # Just re-greedy the row. If it's the same, try again.
                # If we do this many times, we are wasting time.
                
                # Let's try a "Jump": 
                # Set the max element of the row to current_scope + random(1, 10)
                # and try to fill the rest?
                
                # Actually, let's just re-greedy. 
                # If the "forbidden" set changed (because other rows changed), 
                # the greedy result might be different.
                # But we only changed this row.
                
                # So, to force change, we can **randomly remove** one difference from 'used' 
                # that belongs to this row, and add a random large difference? 
                # No, that breaks validity.
                
                # Let's stick to: If no improvement, just continue. 
                # The random selection of rows will eventually find a better configuration.
                
                pass
            
            used.update(p_diffs)
            no_improve_count = 0

    return rows
# EVOLVE-BLOCK-END
