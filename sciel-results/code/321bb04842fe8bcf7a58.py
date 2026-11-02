import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_rows = None
    best_max_val = float('inf')
    
    def get_max_val(rows):
        return max(max(row) for row in rows) if rows else 0

    def is_valid_row(row, used_diffs, exclude_diffs=None):
        """
        Check if a row is valid given current used differences.
        If exclude_diffs is provided, ignore conflicts with those specific diffs (e.g., when replacing an element).
        """
        if exclude_diffs is None:
            exclude_diffs = set()
            
        # Check internal diffs of the row
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False # Duplicates within row (shouldn't happen if increasing)
                row_diffs.add(d)
                
        # Check against global used diffs
        # We must ensure that (row_diffs - exclude_diffs) does not intersect (used_diffs - exclude_diffs)
        # Actually, simpler: 
        # The new set of diffs introduced by this row is row_diffs.
        # The diffs removed from the global pool (if we are replacing) are exclude_diffs.
        # The effective new usage is (row_diffs \ exclude_diffs).
        # The effective existing usage is (used_diffs \ exclude_diffs).
        # They must be disjoint.
        
        new_diffs = row_diffs - exclude_diffs
        existing_diffs = used_diffs - exclude_diffs
        
        if new_diffs & existing_diffs:
            return False
        return True

    def randomized_greedy():
        used_diffs = set()
        rows = []
        # Window size for randomization
        window_size = 5
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates
                candidates = []
                x = row[-1] + 1
                # Limit search space to avoid infinite loops in pathological cases, 
                # though greedy usually finds a solution quickly.
                # A safe upper bound for x in a valid greedy step is roughly current_max * (k+1) or similar.
                # But let's just scan upwards.
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        if len(candidates) >= window_size:
                            break
                    x += 1
                    # Safety break if x gets too large (should not happen for small k)
                    if x > 10000:
                        break
                
                if not candidates:
                    # Fallback: try larger x until found (deterministic)
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            break
                        x += 1
                            
                # Pick randomly from the smallest candidates
                chosen_x = random.choice(candidates)
                row.append(chosen_x)
                used_diffs.update({chosen_x - a for a in row[:-1]})
            
            rows.append(row)
        return rows

    def local_improve(rows):
        """
        Attempt to reduce the maximum value by modifying the largest rows.
        """
        if not rows:
            return rows
            
        # Deep copy to avoid modifying original during exploration if we wanted to, 
        # but we will modify in place for speed and return the best.
        rows = [r[:] for r in rows]
        
        max_iter = 50
        for _ in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Find the row with the largest maximum element
            max_val = max(max(row) for row in rows)
            # Pick one of the rows with this max value
            target_row_idx = -1
            for i, r in enumerate(rows):
                if max(r) == max_val:
                    target_row_idx = i
                    break
            
            if target_row_idx == -1:
                break
                
            target_row = rows[target_row_idx]
            last_elem = target_row[-1]
            
            # Strategy 1: Replace the last element with a smaller one
            # We need to find x < last_elem such that:
            # 1. x > target_row[-2]
            # 2. The set of diffs for the new row (with x) does not conflict with global used diffs, 
            #    EXCEPT for the diffs that were created by last_elem.
            
            # Calculate diffs created by last_elem in the current row
            old_diffs = {last_elem - a for a in target_row[:-1]}
            
            # Current global used diffs
            # We need to reconstruct or maintain a global set. 
            # Maintaining a global set in local_improve is tricky if we modify rows.
            # Let's maintain a global used_diffs set outside.
            
            # For simplicity in this nested function, let's just check validity against all other rows.
            # This is O(n^2 * k^2) per check, which might be slow.
            # Better: maintain a global `used_diffs` set in the outer scope and update it.
            
            # Let's refactor: pass used_diffs as a mutable set.
            pass # Placeholder for logic below

        return rows

    # To properly implement local_improve with global state, we need to manage `used_diffs`.
    # Let's rewrite the main loop to manage state explicitly.

    def solve():
        best_rows = None
        best_score = float('inf')
        
        while time.time() - start_time < time_limit:
            # 1. Generate
            rows, used_diffs = randomized_greedy_with_state()
            
            # 2. Improve
            rows, used_diffs = local_improve_with_state(rows, used_diffs)
            
            # 3. Evaluate
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [r[:] for r in rows]
                
                # Early termination if we hit a known good bound? 
                # For (7,5), best is 112. If we find 112, we can stop? 
                # The task says "best known set has largest entry 112". 
                # If we find 112, it's optimal for this specific instance? 
                # The prompt implies we are solving for generic n_rows, k passed by evaluator.
                # So we shouldn't hardcode 112.
                
            # Optional: If best_score is very low, increase time budget for other runs?
            # No, just keep going.
            
        return best_rows

    def randomized_greedy_with_state():
        used_diffs = set()
        rows = []
        window_size = 5
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                # Scan for candidates
                while len(candidates) < window_size:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                    x += 1
                    if x > 5000: break # Safety
                
                if not candidates:
                    # Fallback: find the very first valid one
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            break
                        x += 1
                            
                chosen_x = random.choice(candidates)
                row.append(chosen_x)
                used_diffs.update({chosen_x - a for a in row[:-1]})
            
            rows.append(row)
        return rows, used_diffs

    def local_improve_with_state(rows, used_diffs):
        # We will try to shrink the largest rows.
        # We need to be careful to update used_diffs correctly.
        
        max_improve_iters = 100
        for _ in range(max_improve_iters):
            if time.time() - start_time > time_limit - 2:
                break
                
            # Find max row
            max_val = max(max(r) for r in rows)
            # Pick a row with this max
            target_idx = None
            for i, r in enumerate(rows):
                if max(r) == max_val:
                    target_idx = i
                    break
            
            if target_idx is None:
                break
                
            target_row = rows[target_idx]
            last_elem = target_row[-1]
            
            # Try to replace last_elem with a smaller x
            # Range: (target_row[-2], last_elem)
            lower_bound = target_row[-2] + 1
            upper_bound = last_elem - 1
            
            if lower_bound > upper_bound:
                continue # Can't shrink last element further without breaking order or range
                
            # Calculate diffs to be removed
            old_diffs = {last_elem - a for a in target_row[:-1]}
            
            # We need to find x such that:
            # 1. x is in [lower_bound, upper_bound]
            # 2. x is not in target_row (it's not, since it's < last_elem and > second last)
            # 3. New diffs {x - a for a in target_row[:-1]} do not conflict with 
            #    (used_diffs - old_diffs)
            
            # Let's pre-calculate the "available" diffs
            # available_diffs = used_diffs - old_diffs
            # We need new_diffs to be disjoint from available_diffs.
            
            # Optimization: Iterate x from upper_bound down to lower_bound?
            # Smaller x is better (reduces max). So iterate from lower_bound up?
            # Actually, any x < last_elem reduces the max of that row to max(target_row[:-1], x).
            # Since x > target_row[-2], the new max is x.
            # So we want the smallest valid x.
            
            found = False
            for x in range(lower_bound, upper_bound + 1):
                new_diffs = {x - a for a in target_row[:-1]}
                
                # Check conflict
                # Conflict if any d in new_diffs is in used_diffs AND d is not in old_diffs
                # Wait, if d is in old_diffs, it's being removed. So it's fine.
                # So we check: new_diffs & (used_diffs - old_diffs) == empty set
                
                # Efficient check:
                conflict = False
                for d in new_diffs:
                    if d in used_diffs and d not in old_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Valid!
                    # Update used_diffs
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    
                    # Update row
                    target_row[-1] = x
                    found = True
                    break
            
            if not found:
                # Try other rows? Or break?
                # To avoid getting stuck, we can try a random row with max value, 
                # or try a different move like swapping.
                # For now, if the max row can't be shrunk, we might be stuck in a local minimum.
                # Let's try to swap the last element of the target row with the last element of another row?
                # That's complex. Let's just break if the current max row can't be improved.
                # But this might stop improvement too early.
                # Let's try a different row with the same max value if it exists.
                improved_any = False
                for i, r in enumerate(rows):
                    if i == target_idx: continue
                    if max(r) == max_val:
                        # Try to improve this one instead
                        # (Logic duplication... let's just break for now to keep code simple, 
                        #  relying on multiple restarts to escape local minima)
                        pass
                break

        return rows, used_diffs

    return solve()
