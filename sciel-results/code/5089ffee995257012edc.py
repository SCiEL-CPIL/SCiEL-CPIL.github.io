import time
import random

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Use 140s for logic.
    deadline = start_time + 140.0
    
    # If parameters are trivial
    if n_rows == 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]

    def get_scope(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)

    def is_valid(rows):
        """
        Checks if the rows are valid (all diffs distinct).
        """
        used = set()
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            used |= diffs
        return True

    def greedy_construct(n_rows, k, randomness=0.0):
        """
        Greedy construction with optional randomization.
        Returns list of rows or None if failed.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                candidates = []
                x = last + 1
                # Limit candidate search to avoid infinite loops or excessive scope
                # We look for a few valid candidates
                while len(candidates) < 5:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                    # If x gets very large, we might be stuck
                    if x > 1000:
                        break
                
                if not candidates:
                    return None
                
                if randomness > 0.0 and len(candidates) > 1:
                    # Pick a random candidate from the top few
                    # Bias towards smaller candidates
                    idx = random.randint(0, min(4, len(candidates) - 1))
                    chosen = candidates[idx]
                else:
                    chosen = candidates[0]
                
                row.append(chosen)
                # Update used diffs
                for a in row[:-1]:
                    used.add(chosen - a)
            
            rows.append(row)
        
        # Final validation
        if is_valid(rows):
            return rows
        else:
            return None

    def rebuild_row(rows, row_idx, used, target_max):
        """
        Attempts to replace rows[row_idx] with a new row that has max < target_max.
        Modifies 'used' and 'rows' in place on success.
        Returns True on success.
        """
        curr_row = rows[row_idx]
        curr_diffs = set()
        for i in range(len(curr_row)):
            for j in range(i + 1, len(curr_row)):
                curr_diffs.add(curr_row[j] - curr_row[i])
        
        # Remove current row's diffs from used
        used -= curr_diffs
        
        # Try to construct a new row
        # We try a few times with randomization
        for _ in range(50):
            row = [0]
            success = True
            while len(row) < k + 1:
                last = row[-1]
                candidates = []
                x = last + 1
                # We must ensure the final max is < target_max
                # So x must be < target_max
                # Also, we need space for remaining elements?
                # Not strictly, as they can be close together, but x < target_max is necessary.
                
                while len(candidates) < 3:
                    if x >= target_max:
                        break
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                    if x > target_max:
                        break
                        
                if not candidates:
                    success = False
                    break
                
                # Pick random candidate
                chosen = random.choice(candidates)
                row.append(chosen)
                for a in row[:-1]:
                    used.add(chosen - a)
            
            if success and max(row) < target_max:
                # Verify validity just in case
                # (The construction ensures no conflict with 'used', and 'used' was updated)
                # But we need to ensure internal diffs are distinct? 
                # The greedy check `new_diffs & used` ensures new diffs don't conflict with previous ones.
                # Does it ensure new diffs are distinct from each other?
                # Yes, because `new_diffs` is a set, and we check intersection with `used`.
                # Wait, `new_diffs` contains diffs of `chosen` against all previous in `row`.
                # These are all distinct from each other? 
                # If row is [0, 2, 5], chosen=7. new_diffs={7,5,2}. Distinct.
                # Generally, if row is increasing, diffs `chosen - a_i` are distinct.
                # So internal validity is maintained.
                
                # Restore used with new diffs
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                used |= new_diffs
                
                rows[row_idx] = row
                return True
        
        # Restore original state
        used |= curr_diffs
        return False

    # Main Logic
    best_rows = None
    best_scope = float('inf')
    
    # 1. Initial Deterministic Greedy
    rows = greedy_construct(n_rows, k, randomness=0.0)
    if rows:
        best_rows = rows
        best_scope = get_scope(rows)
    
    # 2. Optimization Loop
    while time.time() < deadline:
        if not best_rows:
            # Try random restarts
            rows = greedy_construct(n_rows, k, randomness=1.0)
            if rows:
                best_rows = rows
                best_scope = get_scope(rows)
            continue
        
        # Identify the row with the largest maximum element
        max_idx = max(range(n_rows), key=lambda i: best_rows[i][-1])
        current_max = best_rows[max_idx][-1]
        
        # Try to reduce the scope by rebuilding this row
        # Target: one less than current max
        target = current_max - 1
        
        # If target is too small (e.g., < k), we can't improve
        if target < k:
            break
            
        # Compute used set excluding the row to be rebuilt
        used = set()
        for i in range(n_rows):
            if i == max_idx:
                continue
            row = best_rows[i]
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    used.add(row[b] - row[a])
        
        # Attempt rebuild
        if rebuild_row(best_rows, max_idx, used, target):
            new_scope = get_scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                # Continue optimizing with the new best
                continue
            else:
                # This shouldn't happen if target < current_max and rebuild succeeded
                # But if it did, we might have a tie or error. 
                # Since we modified best_rows in place, we should revert if not improved?
                # Actually, if scope didn't decrease, the row changed but max didn't?
                # If max(row) < target < current_max, then new max <= target < current_max.
                # So scope MUST decrease.
                pass
        
        # If local search is not improving, try a full random restart occasionally
        # This helps escape local minima where no single row can be improved
        if random.random() < 0.05:
            rows = greedy_construct(n_rows, k, randomness=1.0)
            if rows:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_rows = rows
                    best_scope = scope

    return best_rows
