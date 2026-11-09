<<<<<<< SEARCH
    def local_search_max_shrink(rows, time_budget):
        """
        Structural Change: Instead of generic Simulated Annealing, use a targeted
        local search that specifically tries to shrink the current maximum element.
        This is more efficient for minimizing the scope.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        used_diffs = set()
        row_diffs = []
        for r in rows:
            rd = compute_row_diffs(r)
            row_diffs.append(rd)
            used_diffs.update(rd)
            
        current_score = best_score
        end_time = time.time() + time_budget
        
        # We will try to shrink the global max repeatedly.
        # If we get stuck, we might need to move other elements to free up space.
        # But first, let's just try to shrink the max.
        
        no_improve = 0
        
        while time.time() < end_time:
            # Find the global max
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_val == 0:
                break
            
            row = rows[max_r]
            current_val = row[max_i]
            
            # Try to find a smaller valid value for this position
            min_val = row[max_i - 1] + 1 if max_i > 0 else 1
            max_allowed = current_val - 1
            if max_i < len(row) - 1:
                max_allowed = min(max_allowed, row[max_i + 1] - 1)
            
            if max_allowed < min_val:
                # Cannot shrink this element. 
                # Strategy: Try to shrink a *different* element to open up space?
                # Or just accept we are stuck and try a different random perturbation?
                # For now, let's try to shrink the *second* largest element?
                # Or just break and let the outer loop try a different seed?
                # Actually, if we can't shrink the max, we are likely at a local minimum for this specific structure.
                # Let's try to move a random element to a slightly different value to "shake" the structure.
                # This is a form of random restart within the local search.
                
                # Pick a random element (not the max)
                r_idx = random.randint(0, n_rows - 1)
                # Avoid index 0
                if k_plus_1 > 1:
                    j_idx = random.randint(1, k)
                else:
                    break
                
                row = rows[r_idx]
                old_val = row[j_idx]
                
                # Try to move it
                low_bound = row[j_idx - 1] + 1
                if j_idx < k:
                    high_bound = row[j_idx + 1] - 1
                else:
                    high_bound = old_val + 10
                
                if low_bound > high_bound:
                    continue
                    
                # Pick a new value
                # Bias towards smaller values if possible
                v = random.randint(low_bound, high_bound)
                
                if v == old_val:
                    continue

                # Check validity
                old_diffs_to_remove = set()
                for i in range(j_idx):
                    old_diffs_to_remove.add(old_val - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    old_diffs_to_remove.add(row[i] - old_val)
                
                new_diffs_to_add = set()
                for i in range(j_idx):
                    new_diffs_to_add.add(v - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    new_diffs_to_add.add(row[i] - v)
                    
                if len(new_diffs_to_add) != k:
                    continue
                
                conflict = False
                for d in new_diffs_to_add:
                    if d in used_diffs:
                        if d not in old_diffs_to_remove:
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                # Apply
                row[j_idx] = v
                used_diffs.difference_update(old_diffs_to_remove)
                used_diffs.update(new_diffs_to_add)
                row_diffs[r_idx] = new_diffs_to_add | (row_diffs[r_idx] - old_diffs_to_remove)
                
                new_score = get_score(rows)
                
                # Always accept if it doesn't increase the score?
                # Or use a simple hill climbing: accept if better, reject if worse.
                # Since we are shaking, we might accept worse temporarily?
                # Let's just accept if it's the same or better, to avoid getting stuck.
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject, undo
                    row[j_idx] = old_val
                    used_diffs.difference_update(new_diffs_to_add)
                    used_diffs.update(old_diffs_to_remove)
                    row_diffs[r_idx] = old_diffs_to_remove | (row_diffs[r_idx] - new_diffs_to_add)
                    no_improve += 1
                    
                if no_improve > 1000:
                    # Stuck, break
                    break
                continue
            
            # If we can shrink the max, try to do so
            # Search for a valid v in [min_val, max_allowed]
            # We want the smallest v.
            # But sometimes a slightly larger v might be better for future moves?
            # Let's just try to find *any* valid v < current_val.
            # To speed up, we can sample or just iterate from min_val up.
            
            found = False
            # Limit search depth to avoid timeout on huge gaps
            # If the gap is huge, iterating is slow.
            # Let's try a few random candidates first, then linear search from bottom?
            # Actually, linear search from min_val is best for minimizing max.
            
            # Optimization: Only search if the gap is small enough.
            # If gap > 500, just pick a random candidate in the middle?
            # No, we want to shrink.
            
            # Let's iterate from min_val to max_allowed, but cap the number of steps.
            # If we don't find one in 1000 steps, we might be stuck.
            
            old_diffs_to_remove = set()
            for i in range(max_i):
                old_diffs_to_remove.add(current_val - row[i])
            for i in range(max_i + 1, k_plus_1):
                old_diffs_to_remove.add(row[i] - current_val)
                
            used_without_old = used_diffs - old_diffs_to_remove
            
            # Try candidates
            # We can't iterate too many.
            # Let's try a random sample of candidates first?
            # Or just linear search from min_val.
            
            # If the range is large, linear search is O(N).
            # Let's try to be smart: 
            # If we just want *any* reduction, we can try the midpoint?
            # But we want the *smallest* max.
            
            # Let's try linear search from min_val, but break if time runs out.
            # Also, we can skip values that are obviously bad?
            # Hard to know.
            
            # Let's just do a limited linear search.
            search_limit = min(max_allowed, min_val + 200)
            
            for v in range(min_val, search_limit + 1):
                new_diffs_to_add = set()
                valid = True
                for i in range(max_i):
                    d = v - row[i]
                    if d in used_without_old:
                        valid = False
                        break
                    new_diffs_to_add.add(d)
                if not valid:
                    continue
                for i in range(max_i + 1, k_plus_1):
                    d = row[i] - v
                    if d in used_without_old:
                        valid = False
                        break
                    new_diffs_to_add.add(d)
                
                if valid and len(new_diffs_to_add) == k:
                    # Found a valid smaller value
                    # Apply
                    row[max_i] = v
                    used_diffs.difference_update(old_diffs_to_remove)
                    used_diffs.update(new_diffs_to_add)
                    row_diffs[max_r] = new_diffs_to_add | (row_diffs[max_r] - old_diffs_to_remove)
                    
                    new_score = get_score(rows)
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    found = True
                    break
            
            if not found:
                # Couldn't shrink the max in the searched range.
                # This triggers the "shake" logic above in the next iteration
                # because max_allowed < min_val is false, but we just failed to find a v.
                # So we need to handle the "failed to shrink" case by shaking.
                # The logic above `if max_allowed < min_val` only handles the case where the range is empty.
                # If the range is not empty but no valid v is found, we are also stuck.
                # Let's add a shake here too.
                
                # Pick a random element to move
                r_idx = random.randint(0, n_rows - 1)
                if k_plus_1 > 1:
                    j_idx = random.randint(1, k)
                else:
                    break
                
                row = rows[r_idx]
                old_val = row[j_idx]
                
                low_bound = row[j_idx - 1] + 1
                if j_idx < k:
                    high_bound = row[j_idx + 1] - 1
                else:
                    high_bound = old_val + 10
                
                if low_bound > high_bound:
                    no_improve += 1
                    continue
                    
                v = random.randint(low_bound, high_bound)
                
                if v == old_val:
                    no_improve += 1
                    continue

                old_diffs_to_remove = set()
                for i in range(j_idx):
                    old_diffs_to_remove.add(old_val - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    old_diffs_to_remove.add(row[i] - old_val)
                
                new_diffs_to_add = set()
                for i in range(j_idx):
                    new_diffs_to_add.add(v - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    new_diffs_to_add.add(row[i] - v)
                    
                if len(new_diffs_to_add) != k:
                    no_improve += 1
                    continue
                
                conflict = False
                for d in new_diffs_to_add:
                    if d in used_diffs:
                        if d not in old_diffs_to_remove:
                            conflict = True
                            break
                
                if conflict:
                    no_improve += 1
                    continue
                
                # Apply
                row[j_idx] = v
                used_diffs.difference_update(old_diffs_to_remove)
                used_diffs.update(new_diffs_to_add)
                row_diffs[r_idx] = new_diffs_to_add | (row_diffs[r_idx] - old_diffs_to_remove)
                
                new_score = get_score(rows)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject, undo
                    row[j_idx] = old_val
                    used_diffs.difference_update(new_diffs_to_add)
                    used_diffs.update(old_diffs_to_remove)
                    row_diffs[r_idx] = old_diffs_to_remove | (row_diffs[r_idx] - new_diffs_to_add)
                    no_improve += 1
                    
                if no_improve > 1000:
                    break

        return best_rows
=======
    def backtracking_search(seed, target_max):
        """
        Structural Change: Use backtracking search with constraint propagation.
        This explores the search space systematically with pruning, which is
        fundamentally different from greedy + local search.
        """
        random.seed(seed)
        used = set()
        rows = []
        # Precompute all possible differences that would be used
        # We'll build rows one by one
        
        def backtrack(row_idx, current_row, pos):
            if time.time() - start_time > time_limit - 10:
                return None
            
            if pos == k_plus_1:
                rows.append(list(current_row))
                return True
            
            if row_idx == n_rows - 1 and pos == k_plus_1 - 1:
                # Last element, last row - check if it fits in target_max
                pass
            
            start_val = current_row[-1] + 1 if pos > 0 else 1
            # For the last position in the last row, we can go up to target_max
            # Otherwise, we need to leave room for remaining elements
            
            if row_idx == n_rows - 1:
                end_val = target_max
            else:
                # Leave room for remaining positions in this row and all subsequent rows
                remaining_positions = (k_plus_1 - pos) + (n_rows - 1 - row_idx) * k_plus_1
                end_val = target_max - remaining_positions + 1
                if end_val < start_val:
                    return None
            
            # Try values in order (increasing) to find the smallest valid solution
            for v in range(start_val, end_val + 1):
                new_diffs = set()
                valid = True
                for i in range(pos):
                    d = v - current_row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check internal consistency (no duplicate diffs within this row)
                if len(new_diffs) != pos:
                    continue
                
                # Try this value
                current_row.append(v)
                used.update(new_diffs)
                
                if pos + 1 < k_plus_1:
                    # Continue with next position in this row
                    if backtrack(row_idx, current_row, pos + 1):
                        return True
                else:
                    # Move to next row
                    if row_idx + 1 < n_rows:
                        if backtrack(row_idx + 1, [0], 0):
                            return True
                    else:
                        # All rows complete
                        return True
                
                # Backtrack
                current_row.pop()
                used.difference_update(new_diffs)
            
            return None
        
        # Try to find a solution with the given target_max
        if backtrack(0, [0], 0):
            return rows
        return None

    def find_min_target_max(seed):
        """
        Binary search for the minimum target_max that allows a valid solution.
        """
        # Lower bound: counting argument
        total_diffs = n_rows * num_diffs_per_row
        # We need m such that m*(m+1)/2 >= total_diffs
        low = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)) - 1
        low = max(low, k)  # At least k for a single row
        
        # Upper bound: from greedy or known result
        high = 200  # Safe upper bound for 4x8 case
        
        # Binary search
        best_rows = None
        while low <= high:
            mid = (low + high) // 2
            rows = backtracking_search(seed, mid)
            if rows is not None:
                best_rows = rows
                high = mid - 1
            else:
                low = mid + 1
            
            if time.time() - start_time > time_limit - 20:
                break
        
        return best_rows
>>>>>>> REPLACE

