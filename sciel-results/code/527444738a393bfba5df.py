import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (5s buffer)
    
    # Total differences per row is k*(k+1)/2
    # For k=7, that's 28 diffs per row.
    # Total diffs = n_rows * 28.
    
    # Helper to get the set of differences for a single row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a configuration is valid
    # Returns True if valid, False otherwise
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            diffs = get_row_diffs(row)
            if len(diffs) != len(row) * (len(row) - 1) // 2:
                return False
            if all_diffs & diffs:
                return False
            all_diffs |= diffs
        return True

    # Greedy construction
    # We construct rows one by one.
    # For each row, we pick elements one by one.
    # To minimize the scope, we want to keep the largest element small.
    # However, we must ensure that the remaining elements can be placed.
    
    def greedy_construct(n_rows, k, randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            current_max = 0
            
            # We need to pick k more elements.
            # Let's try to pick them in increasing order.
            for step in range(k):
                # We want to find the smallest x > current_max such that
                # adding x to the row doesn't conflict with used, and
                # leaves enough room for the remaining (k - step - 1) elements.
                
                # A simple heuristic: just find the smallest valid x.
                # This might lead to a large scope if we get stuck later, 
                # but for small n_rows and k, it's usually okay.
                # To improve, we can look ahead a bit.
                
                # Lower bound for x
                x = current_max + 1
                
                # We'll scan upwards
                # To avoid infinite loops, we set a high limit
                limit = 10000
                
                while x < limit:
                    # Check if x can be added
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        # Check if we can still fit the remaining elements
                        # The remaining elements must be > x.
                        # The smallest possible values for remaining elements are x+1, x+2, ...
                        # This doesn't guarantee validity, but if x is too large, it's bad.
                        
                        # A stronger check: simulate placing the remaining elements greedily
                        # from x+1 upwards and see if we can complete the row.
                        # This is expensive (O(k^3) per step), but k is small.
                        
                        # Let's just take the first valid x for now.
                        # If we get stuck, we might need backtracking.
                        # But for now, let's stick to simple greedy.
                        
                        row.append(x)
                        used |= new_diffs
                        current_max = x
                        break
                    x += 1
                
                if x >= limit:
                    # Failed to find a valid x
                    return None, None
            
            rows.append(row)
        
        return rows, used

    # Local search / Improvement
    # We try to reduce the maximum element of each row.
    
    def local_search(rows, used, deadline):
        # Calculate initial scope
        scope = max(max(r) for r in rows)
        
        # Pre-compute diffs for each row for faster updates
        row_diffs = [get_row_diffs(r) for r in rows]
        
        # used is the union of all row_diffs
        # We can maintain `used` incrementally.
        
        while time.time() < deadline:
            improved = False
            
            # Try to improve each row
            for r_idx in range(n_rows):
                row = rows[r_idx]
                
                # Try to reduce the last element (the max)
                # If we can't reduce the last, maybe reduce a middle element to allow reducing the last?
                # Let's focus on the last element first.
                
                # We want to find a new value for row[-1] that is smaller than current row[-1]
                # and valid.
                
                current_last = row[-1]
                
                # Lower bound for the new last element
                # It must be > row[-2]
                lower_bound = row[-2] + 1
                upper_bound = current_last - 1
                
                if lower_bound > upper_bound:
                    continue
                
                # We scan from lower_bound to upper_bound
                # We prefer smaller values, so scan from lower_bound upwards
                # But we want to find ANY valid value, ideally the smallest.
                
                # To speed up, we can check conflicts quickly.
                # The diffs involving the last element are:
                # {row[-1] - row[i] for i in range(len(row)-1)}
                
                # Let's try values from lower_bound to upper_bound
                found_new_last = False
                
                # Optimization: We can break early if we find a valid value
                # But we want the smallest valid value to maximize future reduction potential.
                
                for new_last in range(lower_bound, upper_bound + 1):
                    # Calculate new diffs involving new_last
                    new_diffs_last = {new_last - a for a in row[:-1]}
                    
                    # Diffs NOT involving the last element remain the same
                    # old_diffs_last = {current_last - a for a in row[:-1]}
                    
                    # The set of all diffs in the row changes:
                    # old_row_diffs = row_diffs[r_idx]
                    # new_row_diffs = (old_row_diffs - old_diffs_last) | new_diffs_last
                    
                    # We need to check if new_diffs_last conflicts with (used - old_diffs_last)
                    # i.e., new_diffs_last should not contain any diff that is in `used` but not in `old_diffs_last`
                    
                    # Let's compute old_diffs_last
                    old_diffs_last = {current_last - a for a in row[:-1]}
                    
                    # Check conflict
                    # Conflict if (new_diffs_last & (used - old_diffs_last)) is not empty
                    # Note: new_diffs_last and old_diffs_last might overlap?
                    # No, because new_last != current_last, and row[:-1] is fixed.
                    # So new_diffs_last and old_diffs_last are disjoint?
                    # Not necessarily. E.g., if row is [0, 5, 10], last=10.
                    # old_diffs: {10, 5}.
                    # If new_last=7, new_diffs: {7, 2}.
                    # They are disjoint here.
                    # Generally, since new_last < current_last, and all a < new_last,
                    # new_last - a < current_last - a.
                    # So yes, they are disjoint.
                    
                    # So we just need to check if new_diffs_last has any intersection with `used`
                    # EXCEPT for the ones that are being removed (old_diffs_last).
                    # But since they are disjoint, we just check if new_diffs_last has any intersection with `used`.
                    # Wait, `used` contains `old_diffs_last`.
                    # If `new_diffs_last` intersects `used`, it might be intersecting with `old_diffs_last`?
                    # No, they are disjoint.
                    # So if `new_diffs_last & used` is not empty, it means it intersects with some other row's diffs
                    # or a diff from this row that is NOT being removed.
                    
                    # So the check is simply:
                    if not (new_diffs_last & used):
                        # Valid!
                        # Update row and used
                        row[-1] = new_last
                        row_diffs[r_idx] = (row_diffs[r_idx] - old_diffs_last) | new_diffs_last
                        used -= old_diffs_last
                        used |= new_diffs_last
                        
                        # Update scope
                        new_scope = max(max(r) for r in rows)
                        if new_scope < scope:
                            scope = new_scope
                        improved = True
                        found_new_last = True
                        break
                
                # If we reduced the last element, we might be able to reduce other elements now?
                # Or at least we made progress.
                
                # We can also try to reduce other elements.
                # For now, let's just loop again if improved.
                if found_new_last:
                    break # Restart the row loop to re-evaluate other rows or this row again
            
            if not improved:
                # No simple reduction of the last element found.
                # Try a "kick" or more complex move?
                # For now, if we can't improve the last element, we might be stuck.
                # We can try to reduce a middle element to create space.
                
                # Let's try to reduce any element j (0 < j < len(row)-1)
                # This is more complex.
                
                any_improved = False
                for r_idx in range(n_rows):
                    row = rows[r_idx]
                    for j in range(1, len(row) - 1): # Don't touch 0 or last
                        current_val = row[j]
                        
                        # Lower bound
                        lower_bound = row[j-1] + 1
                        # Upper bound
                        upper_bound = row[j+1] - 1
                        
                        if lower_bound > upper_bound:
                            continue
                        
                        # Try to find a smaller valid value for row[j]
                        # We scan from lower_bound to upper_bound
                        # We prefer smaller values.
                        
                        # Diffs involving row[j]:
                        # {row[j] - row[i] for i < j} U {row[m] - row[j] for m > j}
                        
                        # Let's just try a few random smaller values to save time
                        # Or scan downwards from current_val - 1?
                        # Scanning upwards from lower_bound finds the smallest valid value, which is best.
                        
                        # But scanning the whole range might be slow.
                        # Let's limit the search.
                        
                        # For now, let's just try to find ANY valid value smaller than current_val.
                        # We can iterate from lower_bound to upper_bound.
                        
                        found = False
                        for new_val in range(lower_bound, upper_bound + 1):
                            if new_val >= current_val:
                                break # We want smaller values, and we are scanning up.
                                # Wait, if we scan from lower_bound up, we find the smallest.
                                # If we find one, it's < current_val?
                                # Not necessarily. lower_bound could be > current_val? No, lower_bound < current_val.
                                # So the first valid one found is the smallest valid one.
                                # Is it smaller than current_val?
                                # Yes, because upper_bound <= current_val - 1 (if j is not last).
                                # Wait, if j is last-1, upper_bound = row[last] - 1.
                                # current_val = row[last-1].
                                # So upper_bound > current_val?
                                # row[last] > row[last-1], so row[last]-1 >= row[last-1].
                                # So upper_bound >= current_val.
                                # So we might find a value >= current_val.
                                # We only want to accept if new_val < current_val.
                                
                                # Actually, if we just want to reduce the scope, reducing a middle element
                                # doesn't directly reduce the scope unless it allows reducing the last element.
                                # So this move is only useful if it leads to a reduction in the last element.
                                # This requires lookahead.
                                
                                # Given the complexity, let's skip this for now and rely on the last-element reduction.
                                # If we get stuck, we might need a more sophisticated search.
                                
                                pass
                            
                            # Calculate new diffs involving new_val
                            new_diffs = set()
                            for i in range(j):
                                new_diffs.add(new_val - row[i])
                            for m in range(j+1, len(row)):
                                new_diffs.add(row[m] - new_val)
                            
                            # Old diffs involving current_val
                            old_diffs = set()
                            for i in range(j):
                                old_diffs.add(current_val - row[i])
                            for m in range(j+1, len(row)):
                                old_diffs.add(row[m] - current_val)
                            
                            # Check conflict
                            # new_diffs must not intersect (used - old_diffs)
                            # i.e., new_diffs & (used - old_diffs) == empty
                            
                            # Note: new_diffs and old_diffs might overlap?
                            # No, because new_val != current_val.
                            
                            # So check if new_diffs has any intersection with used, excluding old_diffs.
                            # Since they are disjoint, just check if new_diffs & (used - old_diffs) is empty.
                            # Which is equivalent to: (new_diffs & used) <= old_diffs.
                            # But since they are disjoint, (new_diffs & used) must be empty.
                            # Wait, `used` contains `old_diffs`.
                            # So if `new_diffs & used` is not empty, it must be intersecting with something else.
                            # So the check is simply: not (new_diffs & used)
                            
                            if not (new_diffs & used):
                                # Valid
                                # Apply move
                                row[j] = new_val
                                row_diffs[r_idx] = (row_diffs[r_idx] - old_diffs) | new_diffs
                                used -= old_diffs
                                used |= new_diffs
                                any_improved = True
                                found = True
                                break
                        
                        if found:
                            break # Move to next row
                    
                    if any_improved:
                        break # Move to next row or restart
                
                if not any_improved:
                    # Stuck
                    # We can try a random restart or a kick.
                    # For now, break to save time.
                    break
            
            # Check time
            if time.time() > deadline:
                break
        
        return rows, scope

    # Main loop
    best_rows = None
    best_scope = float('inf')
    
    # Run multiple greedy starts
    # We can vary the randomization in greedy if we implement it.
    # Currently greedy is deterministic.
    # Let's add some randomness to the greedy construction to explore different structures.
    
    def greedy_construct_random(n_rows, k, rand_prob=0.1):
        used = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            current_max = 0
            
            for step in range(k):
                x = current_max + 1
                limit = 10000
                valid_candidates = []
                
                # Find a few valid candidates
                # We look for the smallest, but sometimes pick a larger one
                # to create different structures.
                
                while x < limit and len(valid_candidates) < 5:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_candidates.append(x)
                    x += 1
                
                if not valid_candidates:
                    return None, None
                
                if random.random() < rand_prob and len(valid_candidates) > 1:
                    # Pick a random candidate
                    # Prefer smaller ones?
                    # Just pick randomly from the found ones.
                    chosen = random.choice(valid_candidates)
                else:
                    chosen = valid_candidates[0]
                
                row.append(chosen)
                # Update used
                new_diffs = {chosen - a for a in row[:-1]}
                used |= new_diffs
                current_max = chosen
            
            rows.append(row)
        
        return rows, used

    attempts = 0
    while time.time() < deadline - 5: # Leave 5s for final optimization
        # Try a few different random seeds / probabilities
        rand_prob = random.uniform(0.0, 0.2)
        rows, used = greedy_construct_random(n_rows, k, rand_prob)
        
        if rows is None:
            continue
        
        # Optimize this solution
        opt_rows, scope = local_search(rows, used, min(time.time() + 5, deadline - 2))
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in opt_rows]
            
            # If we found a very good solution, we can spend more time on it
            # But for now, we just keep searching.
        
        attempts += 1
        # If we've done many attempts, we can stop and focus on the best one
        if attempts > 10:
            # Do a final long optimization on the best solution found so far
            # We need to reconstruct `used` for best_rows
            best_used = set()
            for r in best_rows:
                best_used |= get_row_diffs(r)
            
            final_rows, final_scope = local_search(best_rows, best_used, deadline)
            
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = final_rows
            
            break

    if best_rows is None:
        # Fallback to simple greedy
        rows, used = greedy_construct(n_rows, k)
        if rows:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
    
    return best_rows
