import time
import random
import math

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers per row)
    # We want to minimize M (max entry)
    
    start_time = time.time()
    # Safety margin for the 150s limit. Aim for ~140s of work.
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Randomized greedy for this row to escape local minima in row construction
            # Jitter increases with attempts to explore more
            jitter = 0.12 if attempt > 20 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 100):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                # Verify the whole set is valid with this new row
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        max_passes = 8
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Sort indices by descending max entry to target the largest rows first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=60)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        # Strategy: Rebuild 1 or 2 random rows with high randomness
        num_kicks = random.choice([1, 1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with high jitter
            success = False
            # Try a few times to find a valid row, even if larger
            for attempt in range(30):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    # Wider search range for kicks to allow for larger numbers if needed
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            # High jitter: skip valid candidates often to explore
                            if random.random() < 0.4 and trial_x < 400:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]
                
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # If we failed to rebuild, keep a greedy row
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- High-precision element-wise polish (from inspiration) ----------
    def high_precision_polish(rows):
        """
        Exhaustively try to reduce every non-first element in each row.
        This is a local search over individual positions.
        """
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 5
        pass_count = 0
        
        while improved and time_left() > 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # Process rows in order of their current max
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            
            for r_idx in order:
                if time_left() < 10:
                    break
                    
                row = rows[r_idx]
                # For each position j (from last to first, to maintain prefix validity easily? 
                # Actually, changing a value affects all subsequent diffs. 
                # Let's just try to reduce row[j] for j in range(len(row)-1, 0, -1)
                for j in range(len(row) - 1, 0, -1):
                    if time_left() < 10:
                        break
                    
                    current_val = row[j]
                    # Try to find a smaller value for row[j]
                    # It must be > row[j-1] and < row[j+1] (if exists)
                    # And all new diffs must be valid against used_other and row's other diffs
                    
                    used_other = set()
                    for o_idx in range(n_rows):
                        if o_idx != r_idx:
                            used_other |= row_diffs_set(rows[o_idx])
                    
                    # Compute diffs within the row excluding those involving index j?
                    # No, we need to check all diffs involving j against the rest of the world.
                    
                    # Let's calculate the set of "forbidden" values for row[j]
                    # A value v is forbidden if:
                    # 1. v - row[i] is in used_other for any i < j
                    # 2. row[i] - v is in used_other for any i > j
                    # 3. v - row[i] is in row_diffs_set(row) for any i < j (excluding the one we are replacing? No, we are replacing, so we need to check against the OTHER elements)
                    # 4. row[i] - v is in row_diffs_set(row) for any i > j
                    
                    # Simpler: Try candidate values from current_val - 1 down to row[j-1] + 1
                    # And check validity.
                    
                    best_candidate = None
                    # We want the smallest valid value >= row[j-1] + 1
                    # But we also want it to be as small as possible to reduce scope?
                    # Actually, reducing any element helps. But reducing later elements is often easier.
                    # Let's just try to find ANY valid value < current_val.
                    # To maximize reduction, we should try the smallest possible valid value.
                    
                    min_possible = row[j-1] + 1
                    max_possible = current_val - 1
                    if j < len(row) - 1:
                        max_possible = min(max_possible, row[j+1] - 1)
                    
                    if min_possible > max_possible:
                        continue
                    
                    # Iterate from smallest to largest to find the tightest fit?
                    # Or from largest to smallest?
                    # If we pick a very small value, we might block future reductions?
                    # Let's just scan all candidates in range and pick the smallest one that works.
                    # This is O(Range * Diffs). Range is usually small.
                    
                    for v in range(min_possible, max_possible + 1):
                        # Check validity
                        valid = True
                        new_row = list(row)
                        new_row[j] = v
                        
                        # Check diffs involving j
                        for i in range(len(new_row)):
                            if i == j:
                                continue
                            d = abs(new_row[j] - new_row[i])
                            # Check against other rows
                            if d in used_other:
                                valid = False
                                break
                            # Check against other diffs in the same row (excluding the old ones involving j)
                            # We need to check d against all diffs in the row that do NOT involve index j.
                            # And also check if d duplicates any other diff in the row involving j? No, d is unique for a fixed j and varying i? 
                            # Wait, d = |v - row[i]|. For fixed v and j, d varies with i.
                            # We must ensure that for all i1 != i2, |v - row[i1]| != |v - row[i2]|? 
                            # No, the condition is that ALL differences in the row are distinct.
                            # So if i1 < i2, row[i2]-row[i1] is a diff.
                            # If we change row[j], we change diffs (j, i) for all i != j.
                            # We must ensure:
                            # 1. New diffs (j, i) are not in used_other.
                            # 2. New diffs (j, i) are not in the set of diffs of the row that do NOT involve j.
                            # 3. New diffs (j, i) are distinct from each other. (i.e., |v - row[i1]| != |v - row[i2]| for i1 != i2)
                            
                        if valid:
                            # Now check against the rest of the row's diffs
                            rest_diffs = set()
                            for i in range(len(new_row)):
                                for l in range(i + 1, len(new_row)):
                                    if i != j and l != j:
                                        rest_diffs.add(new_row[l] - new_row[i])
                            
                            new_j_diffs = []
                            for i in range(len(new_row)):
                                if i != j:
                                    d = abs(new_row[j] - new_row[i])
                                    new_j_diffs.append(d)
                                    if d in rest_diffs:
                                        valid = False
                                        break
                                    if d in used_other:
                                        valid = False
                                        break
                            
                            if valid:
                                # Check if new_j_diffs has duplicates
                                if len(set(new_j_diffs)) != len(new_j_diffs):
                                    valid = False
                            
                        if valid:
                            best_candidate = v
                            break # We found the smallest valid v, which is best for reducing scope
                            
                    if best_candidate is not None:
                        row[j] = best_candidate
                        improved = True
                        # Continue to next j or next row? 
                        # Since row changed, the constraints for other j's changed.
                        # It's safer to restart the inner loop for this row? Or just continue?
                        # Let's continue.
        
        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)
            
        # Apply high precision polish
        if time_left() > 10:
            hp_polished = high_precision_polish(best_rows)
            if scope(hp_polished) < best_scope:
                best_rows = hp_polished
                best_scope = scope(best_rows)

    # Iterated Local Search with multiple restarts
    # We use a loop that alternates between Local Search and Kicks
    # Also occasionally restart from scratch with different random seeds
    while time_left() > 25:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            
        # 2. High Precision Polish (cheap and effective)
        if time_left() > 15:
            hp_polished = high_precision_polish(best_rows)
            if scope(hp_polished) < best_scope:
                best_rows = hp_polished
                best_scope = scope(best_rows)

        # 3. Kick
        if time_left() < 20:
            break
            
        kicked_rows = kick(best_rows)
        
        # 4. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 5. High Precision Polish on kicked rows
        if time_left() > 15:
            hp_polished_kicked = high_precision_polish(polished_kicked)
            if scope(hp_polished_kicked) < best_scope:
                best_rows = hp_polished_kicked
                best_scope = scope(best_rows)
            elif scope(polished_kicked) < best_scope:
                best_rows = polished_kicked
                best_scope = scope(best_rows)
        
        # 6. Occasionally do a full restart with fresh greedy + polish
        # This helps escape deep local minima
        if time_left() > 40 and random.random() < 0.15:
            fresh = greedy()
            fresh_polished = local_polish(fresh)
            if time_left() > 10:
                fresh_hp = high_precision_polish(fresh_polished)
                if scope(fresh_hp) < best_scope:
                    best_rows = fresh_hp
                    best_scope = scope(best_rows)
                elif scope(fresh_polished) < best_scope:
                    best_rows = fresh_polished
                    best_scope = scope(best_rows)

    # Final polish pass
    if time_left() > 5:
        final_polished = local_polish(best_rows)
        if scope(final_polished) < best_scope:
            best_rows = final_polished
            best_scope = scope(best_rows)
            
        if time_left() > 3:
            final_hp = high_precision_polish(best_rows)
            if scope(final_hp) < best_scope:
                best_rows = final_hp
                best_scope = scope(best_rows)

    return best_rows
