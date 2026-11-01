"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
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
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def backtrack_row(used_other, max_val=None, timeout=5.0):
        """Backtracking search for a single row given diffs used by other rows.
        Returns a valid row (list of k+1 ints starting at 0) or None."""
        start = time.time()
        if max_val is None:
            max_val = 200  # reasonable upper bound for n_rows=4, k=7
        
        row = [0]
        row_diffs = set()
        
        def bt(pos, current_max):
            nonlocal row, row_diffs
            if time.time() - start > timeout:
                return False
            if pos == k + 1:
                return True
            
            # Try candidates from current_max+1 upward, but also try smaller
            # values to find tighter rows
            start_candidate = current_max + 1
            for cand in range(start_candidate, max_val + 1):
                # Check differences
                new_diffs = {cand - row[i] for i in range(pos)}
                if new_diffs & used_other or new_diffs & row_diffs:
                    continue
                # Check internal consistency (all pairwise diffs distinct)
                # Already checked above since row_diffs contains all prior diffs
                
                row.append(cand)
                row_diffs |= new_diffs
                
                if bt(pos + 1, cand):
                    return True
                
                row.pop()
                row_diffs -= new_diffs
            
            return False
        
        if bt(1, 0):
            return row
        return None

    def try_reduce_row(rows, row_idx, used_other, max_attempts=150):
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
            jitter = 0.15 if attempt > 15 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 80):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 250:
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
        # if we are in an ILS loop
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
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
        # Strategy: Rebuild 1 or 2 random rows, with a bias toward the largest rows
        num_kicks = random.choice([1, 2])
        # Weight selection toward rows with larger max entries
        weights = [max(rows[i]) + 1 for i in range(n_rows)]
        total_w = sum(weights)
        indices = []
        remaining = list(range(n_rows))
        for _ in range(num_kicks):
            r = random.random() * total_w
            cum = 0
            chosen = remaining[0]
            for idx in remaining:
                cum += weights[idx]
                if r <= cum:
                    chosen = idx
                    break
            indices.append(chosen)
            remaining.remove(chosen)
            total_w -= weights[chosen]
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Try backtracking first for a tighter row
            bt_row = backtrack_row(used_other, max_val=180, timeout=3.0)
            if bt_row is not None:
                rows[idx] = bt_row
                continue
            
            # Fall back to randomized greedy with jitter
            success = False
            for attempt in range(20):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                jitter = random.uniform(0.2, 0.5)
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < jitter and trial_x < 300:
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
                # Greedy fallback
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

    # New: High-precision polish that tries individual element adjustments
    def high_precision_polish(rows, max_iterations=100):
        """Try to reduce scope by adjusting individual elements, including increases."""
        rows = [list(r) for r in rows]
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time_left() > 3:
            improved = False
            iterations += 1
            
            # Process rows in order of largest max entry
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            
            for row_idx in order:
                if time_left() < 3:
                    break
                
                row = rows[row_idx]
                used_other = set()
                for j in range(n_rows):
                    if j != row_idx:
                        used_other |= row_diffs_set(rows[j])
                
                # Try to reduce each element (except the first, which is 0)
                for pos in range(1, len(row)):
                    if time_left() < 2:
                        break
                    
                    old_val = row[pos]
                    
                    # Try decreasing the value (primary goal)
                    for delta in range(1, min(15, old_val)):
                        new_val = old_val - delta
                        if new_val <= row[pos-1]:
                            break
                        
                        new_row = list(row)
                        new_row[pos] = new_val
                        
                        # Check all differences are distinct
                        new_diffs = set()
                        valid = True
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                d = new_row[j] - new_row[i]
                                if d in new_diffs or d in used_other:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if not valid:
                                break
                        
                        if valid:
                            rows[row_idx] = new_row
                            improved = True
                            break  # Move to next element
                    
                    # Also try increasing middle elements slightly to enable
                    # later reductions (this can help escape local minima)
                    if pos < len(row) - 1 and time_left() > 4:
                        for delta in range(1, 5):
                            new_val = old_val + delta
                            if new_val >= row[pos+1]:
                                break
                            
                            new_row = list(row)
                            new_row[pos] = new_val
                            
                            new_diffs = set()
                            valid = True
                            for i in range(len(new_row)):
                                for j in range(i + 1, len(new_row)):
                                    d = new_row[j] - new_row[i]
                                    if d in new_diffs or d in used_other:
                                        valid = False
                                        break
                                    new_diffs.add(d)
                                if not valid:
                                    break
                            
                            if valid and max(new_row) < max(rows[row_idx]):
                                # Only accept if it actually reduces scope
                                # (increasing an element can reduce max if it's the last)
                                rows[row_idx] = new_row
                                improved = True
                                break
        
        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
    
    # New: Apply high-precision polish early
    if time_left() > 8:
        polished_hp = high_precision_polish(best_rows, max_iterations=50)
        if scope(polished_hp) < best_scope:
            best_rows = polished_hp
            best_scope = scope(polished_hp)

    # Iterated Local Search
    # We use a loop that alternates between Local Search and Kicks
    # This is structurally different from a single long SA run.
    while time_left() > 25:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # New: Apply high-precision polish after local polish
        if time_left() > 18:
            polished_hp = high_precision_polish(best_rows, max_iterations=40)
            if scope(polished_hp) < best_scope:
                best_rows = polished_hp
                best_scope = scope(polished_hp)
        
        # 2. Kick
        if time_left() < 18:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # New: Apply high-precision polish on kicked rows
        if time_left() > 12:
            polished_kicked_hp = high_precision_polish(polished_kicked, max_iterations=40)
            if scope(polished_kicked_hp) < best_scope:
                best_rows = polished_kicked_hp
                best_scope = scope(polished_kicked_hp)
            elif scope(polished_kicked) < best_scope:
                best_rows = polished_kicked
                best_scope = scope(polished_kicked)
        else:
            # 4. Update best
            if scope(polished_kicked) < best_scope:
                best_rows = polished_kicked
                best_scope = scope(polished_kicked)

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 8:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
    
    # New: Final high-precision polish with more iterations
    if time_left() > 5:
        polished_hp = high_precision_polish(best_rows, max_iterations=80)
        if scope(polished_hp) < best_scope:
            best_rows = polished_hp
            best_scope = scope(polished_hp)
    
    # Try backtracking on the worst row one more time
    if time_left() > 3:
        worst_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
        used_other = set()
        for j in range(n_rows):
            if j != worst_idx:
                used_other |= row_diffs_set(best_rows[j])
        bt_row = backtrack_row(used_other, max_val=160, timeout=2.0)
        if bt_row is not None:
            candidate = [list(r) for r in best_rows]
            candidate[worst_idx] = bt_row
            if is_valid(candidate) and scope(candidate) < best_scope:
                best_rows = candidate
                best_scope = scope(best_rows)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
