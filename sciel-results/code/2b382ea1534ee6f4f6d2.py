import time
import random
import math
import sys

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                all_diffs.append(row[j] - ri)
    return len(all_diffs) == len(set(all_diffs))

def _scope(rows):
    return max(r[-1] for r in rows)

def _greedy_fallback(n_rows, num_elements):
    """Simple sequential greedy construction as a fallback."""
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy: CSP with Minimum Remaining Values (MRV) heuristic
    # We try to construct rows with a tight scope.
    # For each position, we choose the candidate value that leaves the most
    # "free" difference slots for the rest of the row and subsequent rows.
    
    # Heuristic score: For a candidate value v at position p in row r,
    # we estimate how many future difference values will be blocked.
    # A lower score is better (less blocking).
    
    def _heuristic_score(row_prefix, pos, val, used_mask, num_elements):
        """
        Score a candidate value. Lower is better.
        We want to minimize the number of future difference collisions.
        """
        # 1. Immediate collisions with used_mask are already checked (val is valid)
        # 2. Collisions within the current row (with existing elements)
        #    The differences (val - a) for a in row_prefix are now used.
        #    This is fixed for the choice.
        
        # 3. Future impact: How many potential differences for positions > pos are blocked?
        #    For each future position p' > pos, and each existing element a in row_prefix + [val],
        #    the difference (x - a) for x > val cannot be in used_mask or in row_diffs.
        
        # A simpler proxy: Count how many values in the range [val+1, max_val] 
        # are "blocked" by the new differences introduced by val.
        
        # The new differences are {val - a for a in row_prefix}.
        # Let D_new be this set.
        # For a future element x, the difference x - a is blocked if (x - a) in used_mask | D_new | {existing row diffs}.
        
        # Actually, a better heuristic for "leaving options" is to look at the 
        # number of valid candidates for the *next* position.
        
        next_pos = pos + 1
        if next_pos >= num_elements:
            return 0 # Last element, no future impact within row
        
        # Count valid candidates for next position
        # This is expensive if done fully, so we approximate.
        # We count how many v in [val+1, val+10] are valid for the next step.
        # If val is small, there are more options.
        
        # Let's use a simpler metric: The value itself. Smaller values generally leave more room.
        # But we must avoid "clumping" which causes local diff collisions.
        
        # Let's estimate the number of "free" differences in the range [1, max_val] 
        # that are NOT in (used_mask | row_diffs_so_far | new_diffs).
        
        # This is equivalent to: max_val - popcount(used_mask | row_diffs_so_far | new_diffs).
        # Since used_mask and row_diffs_so_far are constant for this choice, 
        # we just want to minimize popcount(new_diffs) and avoid overlap with used_mask.
        # But new_diffs are determined by val.
        
        # Let's count the number of bits set in new_diffs.
        new_diffs_mask = 0
        for a in row_prefix:
            d = val - a
            new_diffs_mask |= (1 << d)
        
        # If new_diffs overlap with used_mask, it's invalid (already checked).
        # So we want to minimize the number of bits in new_diffs_mask?
        # No, the size of new_diffs_mask is fixed by len(row_prefix).
        
        # The real constraint is global. 
        # Let's use a "density" heuristic.
        # We prefer values that are not too close to each other to keep differences distinct?
        # Actually, distinct differences within a row are required.
        
        # Let's stick to the MRV idea: 
        # Score = - (number of valid candidates for the NEXT position)
        
        # To speed this up, we don't count all, just a sample or a bound.
        # For now, let's just return val. Smaller is better.
        # This effectively becomes a greedy "pick smallest" with backtracking.
        # To make it smarter, we can penalize values that create "small" differences 
        # that are likely to be needed later? No, small differences are "scarce".
        
        # Let's try: Score = val * 10 + (number of small differences created)
        # Small differences are more likely to conflict.
        
        small_diff_count = 0
        for a in row_prefix:
            d = val - a
            if d < 10:
                small_diff_count += 1
                
        return val * 100 + small_diff_count

    def _csp_construct(max_val, rng):
        """
        CSP backtracking with MRV-inspired heuristic.
        """
        rows = []
        used_mask = 0
        
        for row_idx in range(n_rows):
            row = [0]
            # Build row
            success = _build_row_csp(row, used_mask, max_val, rng, rows, num_elements)
            if not success:
                return None
            
            rows.append(row)
            used_mask |= _row_mask(row)
            
        return rows

    def _build_row_csp(row, used_mask, max_val, rng, all_rows, num_elements):
        """
        Recursively build a single row.
        """
        if len(row) == num_elements:
            return True
        
        last_val = row[-1]
        pos = len(row)
        
        # Generate candidates
        candidates = []
        for val in range(last_val + 1, max_val + 1):
            # Check validity
            new_diffs = set()
            valid = True
            for prev in row:
                d = val - prev
                if d in new_diffs:
                    valid = False
                    break
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                # Calculate heuristic score
                # We want to minimize future conflicts.
                # A proxy: How many values in the *next* position are blocked?
                # We approximate by checking a few nearby values for the next step.
                # But that's expensive.
                
                # Alternative: Just use val.
                # To add diversity, we can shuffle candidates slightly.
                candidates.append(val)
        
        if not candidates:
            return False
        
        # Sort by heuristic (smallest first) but add some randomness
        # To avoid getting stuck in local minima, we can shuffle the first few
        if len(candidates) > 5:
            # Take the first 10 and shuffle them, then append the rest
            head = candidates[:10]
            rng.shuffle(head)
            candidates = head + candidates[10:]
        
        for val in candidates:
            row.append(val)
            # Update used_mask for this step? No, used_mask is global.
            # But we need to ensure intra-row uniqueness.
            # We checked new_diffs against each other in the candidate generation.
            
            # Recurse
            if _build_row_csp(row, used_mask, max_val, rng, all_rows, num_elements):
                return True
            
            row.pop()
            
        return False

    # Try to find a solution with scope close to the counting bound
    # We use the CSP approach with different scope limits.
    
    # Determine a good range for max_val
    # The counting bound is a lower bound.
    # We try max_val from counting_bound to counting_bound + 50.
    
    # To save time, we'll try a few specific values.
    attempts = []
    for i in range(0, 60, 2):
        attempts.append(counting_bound + i)
    
    # Shuffle attempts to explore different scopes? No, we want the smallest.
    # But maybe a slightly larger scope is easier to construct?
    # Let's try the smallest first.
    
    for target_scope in attempts:
        if time.time() > DEADLINE - 30:
            break
            
        # Try a few seeds for this scope
        for seed in range(3):
            if time.time() > DEADLINE - 30:
                break
            
            rng_local = random.Random(seed * 1000 + target_scope)
            rows = _csp_construct(target_scope, rng_local)
            
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                    
                # If we found a solution at the bound, we can stop
                if s <= counting_bound + 2:
                    break
        
        if best_rows is not None and best_scope <= counting_bound + 5:
            break
    
    # Fallback if CSP failed or didn't find a good one
    if best_rows is None or best_scope > counting_bound + 20:
        # Use the greedy fallback from the original program
        for seed in range(5):
            if time.time() > DEADLINE - 20:
                break
            rng_local = random.Random(seed * 31 + 17)
            
            rows = []
            used_mask = 0
            success = True
            
            for row_idx in range(n_rows):
                row = [0]
                for pos in range(1, num_elements):
                    found = False
                    # Try values in randomized order
                    candidates = list(range(row[-1] + 1, row[-1] + 50))
                    rng_local.shuffle(candidates)
                    
                    for val in candidates:
                        new_diffs = set()
                        valid = True
                        for prev in row:
                            d = val - prev
                            if d in new_diffs:
                                valid = False
                                break
                            if (used_mask >> d) & 1:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            row.append(val)
                            for d in new_diffs:
                                used_mask |= (1 << d)
                            found = True
                            break
                    
                    if not found:
                        # Fallback: find the smallest valid value
                        val = row[-1] + 1
                        while True:
                            new_diffs = set()
                            valid = True
                            for prev in row:
                                d = val - prev
                                if d in new_diffs or (used_mask >> d) & 1:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                row.append(val)
                                for d in new_diffs:
                                    used_mask |= (1 << d)
                                break
                            val += 1
                
                rows.append(row)
            
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows

    if best_rows is None:
        best_rows = _greedy_fallback(n_rows, num_elements)
        best_scope = _scope(best_rows)
    
    # Local search refinement (simplified from original)
    if time.time() < DEADLINE - 10:
        # Try to reduce the scope by moving elements down
        # We'll do a simple random hill climbing on the scope
        current_rows = [r[:] for r in best_rows]
        current_scope = _scope(current_rows)
        improved = True
        max_iters = 10000
        
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for _ in range(max_iters):
                if time.time() > DEADLINE - 5:
                    break
                
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, num_elements - 1)
                row = current_rows[row_idx]
                old_val = row[pos]
                
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < num_elements - 1 else current_scope
                
                if low >= old_val:
                    continue
                
                new_val = random.randint(low, old_val - 1)
                
                # Check validity
                row[pos] = new_val
                diffs_in_row = set()
                valid = True
                n = num_elements
                for i in range(n):
                    for j in range(i + 1, n):
                        d = row[j] - row[i]
                        if d in diffs_in_row:
                            valid = False
                            break
                        diffs_in_row.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_row_mask = 0
                    for d in diffs_in_row:
                        new_row_mask |= (1 << d)
                    
                    # Check against other rows
                    other_mask = 0
                    for r_idx, r in enumerate(current_rows):
                        if r_idx == row_idx:
                            continue
                        other_mask |= _row_mask(r)
                    
                    if new_row_mask & other_mask:
                        valid = False
                
                if not valid:
                    row[pos] = old_val
                    continue
                
                # Accept
                new_scope = _scope(current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    improved = True
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Revert if scope didn't improve (or even if it did, but we want to keep the best)
                    # Actually, if scope didn't improve, we might still want to accept it to escape local minima?
                    # No, we are minimizing scope. If scope doesn't decrease, we revert.
                    row[pos] = old_val

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return _greedy_fallback(n_rows, num_elements)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
