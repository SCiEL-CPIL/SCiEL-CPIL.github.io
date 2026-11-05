"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # 1. Try column-wise greedy construction with different random seeds
    best_rows = None
    best_scope = float('inf')
    
    # We try a few random orderings of rows in the column-wise construction
    for attempt in range(10):
        if time.time() > DEADLINE - 30.0:
            break
        rows, scope = _column_wise_greedy(n_rows, k, seed=attempt)
        if rows and scope < best_scope:
            best_rows = rows
            best_scope = scope
            
    # If greedy failed or is bad, fall back to a simpler greedy (though column-wise should always work)
    if not best_rows:
        best_rows, best_scope = _simple_greedy(n_rows, k)

    # 2. Local Search / Simulated Annealing to reduce scope
    if time.time() < DEADLINE - 5.0:
        best_rows, best_scope = _local_search_shrink(best_rows, n_rows, k, best_scope, start_time, DEADLINE)

    return best_rows

def _simple_greedy(n_rows, k):
    """Fallback: standard greedy row by row."""
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check if adding x creates conflicts
            # Diffs are x - a for a in row
            conflict = False
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                # Check internal conflicts (diffs within the new row)
                # Since we are adding x, new diffs are x - a.
                # If x - a == b - c for existing a,b,c in row? 
                # Actually, if we just check against 'used_mask', we need to ensure 
                # the new diffs don't collide with EACH OTHER.
                # Since row is increasing, x-a are distinct for distinct a.
                # So we just need to check against used_mask.
                # But wait, if we add x, the new diffs are {x-a | a in row}.
                # These are all distinct from each other.
                # So we just check intersection with used_mask.
                new_diffs_mask = 0
                for a in row:
                    new_diffs_mask |= (1 << (x - a))
                if not (new_diffs_mask & used_mask):
                    row.append(x)
                    used_mask |= new_diffs_mask
        rows.append(row)
    return rows, max(max(r) for r in rows)

def _column_wise_greedy(n_rows, k, seed=0):
    """
    Construct rows by filling columns (positions 0..k).
    Row i has values x[i][0]..x[i][k]. x[i][0]=0.
    We decide x[i][j] for j=1..k.
    Constraint: x[i][j] > x[i][j-1].
    Constraint: All differences x[i][j]-x[i][m] (j>m) are unique across all rows.
    """
    rng = random.Random(seed)
    
    # To break symmetry, we can shuffle the order in which we assign values to rows
    # But actually, we assign all rows for column 1, then all for column 2, etc.
    # We want to keep values small.
    
    # State:
    # rows[i][j] is the value.
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    used_mask = 0
    
    # For each column j from 1 to k
    for j in range(1, k + 1):
        # We need to assign a value to each row i such that:
        # 1. x[i][j] > x[i][j-1]
        # 2. The new differences introduced (x[i][j] - x[i][m] for m < j)
        #    are not in used_mask and do not conflict with each other (across rows).
        
        # To minimize scope, we should try to assign small values.
        # However, the rows are coupled by the global used_mask.
        # Strategy: Try to assign values in increasing order of "difficulty" or just iterate rows?
        # Let's iterate rows, but try to find the smallest valid x[i][j] for each.
        # This is a greedy approach.
        
        # We might need to backtrack if greedy fails. 
        # For now, let's try a randomized greedy: 
        # Pick a random row, find smallest valid x, assign it, update mask. Repeat.
        
        # Actually, a better heuristic: 
        # Sort rows by their current last value (x[i][j-1])? 
        # Or just process them in a fixed order but try multiple candidates?
        
        # Let's try: For each row, find the smallest x > x[i][j-1] that is valid.
        # If we do this sequentially, earlier rows get smaller numbers.
        # To be fair, we can shuffle the order of rows for each column.
        
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        # We need to track which rows have been assigned for this column
        assigned = [False] * n_rows
        
        # We will try to assign values. 
        # A simple loop might fail if an early choice blocks a later row.
        # Backtracking is expensive. 
        # Let's try a "relaxed" greedy: 
        # Find a valid x for each row independently? No, they share used_mask.
        
        # Let's try to assign the smallest possible value to each row in the shuffled order.
        # If it fails, we might need to restart or use a different seed.
        
        # Optimization: Pre-compute potential diffs? No, x is dynamic.
        
        for i in row_order:
            prev_val = rows[i][j-1]
            # Find smallest x > prev_val
            x = prev_val + 1
            while True:
                # Check validity of x for row i
                valid = True
                new_diffs_mask = 0
                for m in range(j):
                    d = x - rows[i][m]
                    # d must be > 0, which is true since x > rows[i][j-1] >= rows[i][m]
                    # Check if d is already used
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    # Check if new_diffs_mask conflicts with itself? No, diffs are distinct for fixed x.
                    # Check if new_diffs_mask conflicts with used_mask? Already done.
                    # Assign
                    rows[i][j] = x
                    used_mask |= new_diffs_mask
                    assigned[i] = True
                    break
                else:
                    x += 1
                    # Safety break to prevent infinite loop if no solution (shouldn't happen for small k)
                    if x > 2000: 
                        return None, float('inf')
                        
    scope = max(max(row) for row in rows)
    return rows, scope

def _local_search_shrink(rows, n_rows, k, initial_scope, start_time, deadline):
    """
    Try to reduce the scope by moving the maximum element of the row containing it
    to a smaller value, or by swapping elements.
    """
    best_rows = [list(r) for r in rows]
    best_scope = initial_scope
    
    def get_used_mask(rows):
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
        return mask
    
    current_mask = get_used_mask(best_rows)
    
    # Simulated Annealing / Hill Climbing
    # We want to minimize max(rows[i][k]).
    
    # Heuristic: Find the row with the max last element.
    # Try to move that last element to a smaller position.
    # Or try to move any element in that row to a smaller position if it doesn't violate order.
    
    # Let's try a simple local search:
    # 1. Identify row r with max last element.
    # 2. Try to decrease rows[r][k] to a smaller value v.
    # 3. Check if v > rows[r][k-1] and if new diffs (v - rows[r][m] for m<k) are valid.
    
    # To allow exploration, we can also try "swaps" or "shifts".
    # But for now, just shrinking the max element.
    
    while time.time() < deadline - 2.0:
        # Find max row
        max_row_idx = 0
        max_val = -1
        for i in range(n_rows):
            if best_rows[i][k] > max_val:
                max_val = best_rows[i][k]
                max_row_idx = i
                
        if max_val <= best_scope: # Should be equal
             pass
        
        # Try to reduce max_val
        row = best_rows[max_row_idx]
        prev_val = row[k-1]
        
        # Try values from prev_val + 1 up to max_val - 1
        # We want the smallest possible, so iterate upwards from prev_val+1
        # But we want to REDUCE the scope, so we are looking for a valid v < max_val.
        # If we find one, we replace max_val with v.
        
        found_better = False
        # We can try all valid v in (prev_val, max_val)
        # To be efficient, we can just check if there is ANY valid v < max_val.
        # If we find the smallest one, we set it.
        
        # Optimization: 
        # The set of diffs for the last element is {v - row[m] for m in 0..k-1}.
        # These must not be in current_mask (excluding the diffs contributed by the old max_val).
        
        # Let's compute the mask of diffs contributed by the OLD last element
        old_diffs_mask = 0
        for m in range(k):
            d = max_val - row[m]
            old_diffs_mask |= (1 << d)
            
        # The "free" mask is current_mask ^ old_diffs_mask (conceptually, though bits might overlap if other rows have same diffs, 
        # but wait: all diffs are UNIQUE across all rows. 
        # So old_diffs_mask bits are ONLY set by this row's last element diffs.
        # Thus, removing them frees those differences.
        
        free_mask = current_mask & ~old_diffs_mask
        
        # Now find smallest v > prev_val such that {v - row[m]} are all in free_mask (i.e., not in free_mask? 
        # No, they must NOT be in the mask of OTHER diffs.
        # The mask of other diffs is free_mask.
        # So we need v such that (1 << (v - row[m])) is NOT set in free_mask for all m.
        
        # Let's iterate v from prev_val + 1 to max_val - 1
        # If we find a valid v, we update.
        
        # To speed up, we can just check a few candidates? 
        # Or iterate. k is small (5). n_rows is small (7).
        
        # Let's iterate.
        for v in range(prev_val + 1, max_val):
            valid = True
            new_diffs_mask = 0
            for m in range(k):
                d = v - row[m]
                if d <= 0: continue # Should be > 0
                # Check if d is in free_mask (i.e., used by someone else)
                if (free_mask >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)
            
            if valid:
                # Also need to check if new_diffs_mask has internal collisions?
                # d's are v - row[m]. Since row[m] are distinct, d's are distinct.
                # So no internal collision.
                
                # Update
                old_last = row[k]
                row[k] = v
                current_mask = (current_mask & ~old_diffs_mask) | new_diffs_mask
                new_scope = max(max(r) for r in best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    found_better = True
                break # Found the smallest valid v, no need to check larger v for this row
            
        if not found_better:
            # Try a different row? Or try to move an earlier element?
            # For now, if we can't shrink the max, we might be stuck.
            # We can try to "shuffle" the rows to break symmetry?
            # Or try to move a non-max element to free up space?
            
            # Let's try to move the second to last element of the max row to a smaller value?
            # This is complex. 
            # Instead, let's try a randomized perturbation: 
            # Pick a random row, pick a random position j, try to move it to a smaller valid position.
            
            if random.random() < 0.5:
                # Try to shrink a random row's last element
                r_idx = random.randint(0, n_rows-1)
                r = best_rows[r_idx]
                last_val = r[k]
                prev_val = r[k-1]
                
                old_diffs_mask = 0
                for m in range(k):
                    d = last_val - r[m]
                    old_diffs_mask |= (1 << d)
                free_mask = current_mask & ~old_diffs_mask
                
                # Try to find v < last_val
                # To allow "uphill" moves (temporary increase in scope) is not helpful for minimizing scope.
                # We only accept if scope decreases.
                # But what if moving last_val UP allows moving another row's max DOWN?
                # That's a swap. 
                
                # Let's just try to find a smaller valid v.
                for v in range(prev_val + 1, last_val):
                    valid = True
                    new_diffs_mask = 0
                    for m in range(k):
                        d = v - r[m]
                        if (free_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs_mask |= (1 << d)
                    if valid:
                        # Accept only if it reduces global scope?
                        # If r is not the max row, reducing r[k] doesn't reduce global scope unless r[k] WAS the max.
                        # But it might free up space for the max row.
                        # Let's accept if it's valid, and check scope.
                        old_last = r[k]
                        r[k] = v
                        current_mask = (current_mask & ~old_diffs_mask) | new_diffs_mask
                        new_scope = max(max(x) for x in best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        # If new_scope >= best_scope, we might want to revert if it doesn't help?
                        # But it might help in future iterations.
                        # Let's keep it if it's valid.
                        break
                        
            # If we are stuck, we might need to restart or do a more complex move.
            # For now, we just break if we can't improve and time is running out?
            # No, we loop until deadline.
            # But if we make no progress, we might be in a local minimum.
            # Let's add a "kick": randomly change a value and see if we can fix it?
            # Too risky.
            
            # Let's try to swap the last elements of two rows?
            # If row A has max 100, row B has max 90.
            # Can we swap 100 and 90?
            # Check if 90 is valid for A and 100 is valid for B.
            
            # Find top 2 rows by last element
            indices = list(range(n_rows))
            indices.sort(key=lambda i: best_rows[i][k], reverse=True)
            i1, i2 = indices[0], indices[1]
            
            r1, r2 = best_rows[i1], best_rows[i2]
            v1, v2 = r1[k], r2[k]
            
            if v1 == v2: continue # Can't swap if equal
            
            # Check if v2 is valid for r1 (replacing v1)
            # And v1 is valid for r2 (replacing v2)
            
            # Diffs for r1 with v1: D1
            # Diffs for r2 with v2: D2
            # Other diffs: Rest
            
            # We need to check if v2's diffs for r1 are in Rest.
            # And v1's diffs for r2 are in Rest.
            
            # Compute D1, D2
            D1 = 0
            for m in range(k):
                D1 |= (1 << (v1 - r1[m]))
            D2 = 0
            for m in range(k):
                D2 |= (1 << (v2 - r2[m]))
            
            Rest = current_mask & ~D1 & ~D2
            
            # Check v2 for r1
            valid1 = True
            D1_new = 0
            for m in range(k):
                d = v2 - r1[m]
                if d <= 0:
                    valid1 = False
                    break
                if (Rest >> d) & 1:
                    valid1 = False
                    break
                D1_new |= (1 << d)
            
            # Check v1 for r2
            valid2 = True
            D2_new = 0
            for m in range(k):
                d = v1 - r2[m]
                if d <= 0:
                    valid2 = False
                    break
                if (Rest >> d) & 1:
                    valid2 = False
                    break
                D2_new |= (1 << d)
            
            # Also check if D1_new and D2_new conflict with each other?
            # They are from different rows, so they must be disjoint.
            if valid1 and valid2 and not (D1_new & D2_new):
                # Perform swap
                r1[k] = v2
                r2[k] = v1
                current_mask = (current_mask & ~D1 & ~D2) | D1_new | D2_new
                new_scope = max(max(x) for x in best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope

    return best_rows, best_scope

# EVOLVE-BLOCK-END
