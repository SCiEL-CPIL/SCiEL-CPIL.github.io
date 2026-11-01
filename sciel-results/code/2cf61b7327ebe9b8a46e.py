"""Combined program: ILS with enhanced backtracking (MRV) and multi-row optimization."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
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
    def greedy(seed=0):
        """
        Randomized greedy construction.
        If seed > 0, introduces jitter to explore different solutions.
        """
        if seed > 0:
            random.seed(seed)
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                window = 100 if (seed > 0 and r_idx > 0) else 50
                for trial_x in range(x + 1, x + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if seed > 0 and random.random() < 0.1 and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(x + 1, x + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    break
                x = row[-1]
            if len(row) == k + 1:
                rows.append(row)
            else:
                pass
        
        if len(rows) != n_rows:
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
            jitter = 0.15 if attempt > 15 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 80):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
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
        max_passes = 8
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
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=100)
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
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            row, x = [0], 0
            current_row_diffs = set()
            success = False
            for attempt in range(25):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 120):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < 0.45 and trial_x < 350:
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

    # ---------- Enhanced Backtracking with MRV ----------
    def backtrack_mrv(target_scope, time_limit):
        """
        Backtracking with MRV heuristic: always expand the position with fewest valid candidates.
        """
        start = time.time()
        
        # State: rows under construction, used diffs
        # We build all n_rows rows, each with k+1 elements starting at 0
        
        rows = [[0] for _ in range(n_rows)]  # each row starts with [0]
        row_diffs = [set() for _ in range(n_rows)]
        used_diffs = set()
        
        # Position to fill: (row_idx, current length)
        # We track which positions are "open" (not yet filled to k+1)
        
        def get_open_positions():
            """Get all open positions (row_idx, next_index_to_fill)."""
            positions = []
            for r in range(n_rows):
                if len(rows[r]) < k + 1:
                    positions.append((r, len(rows[r])))
            return positions
        
        def count_valid_candidates(row_idx, next_idx, val):
            """Check if val is a valid candidate for rows[row_idx][next_idx]."""
            if next_idx > 0 and val <= rows[row_idx][next_idx - 1]:
                return 0  # not increasing
            if val > target_scope:
                return 0
            new_diffs = set()
            for a in rows[row_idx]:
                d = val - a
                if d in used_diffs or d in row_diffs[row_idx]:
                    return 0
                new_diffs.add(d)
            return 1  # valid
        
        def try_mrv_step():
            """Try to fill one position using MRV heuristic."""
            open_pos = get_open_positions()
            if not open_pos:
                return True  # all positions filled
            
            # For each open position, count valid candidates in a small window
            best_pos = None
            best_count = float('inf')
            best_candidates = []
            
            for (r, idx) in open_pos:
                last_val = rows[r][-1]
                # Count valid candidates in range [last_val+1, min(last_val+50, target_scope)]
                count = 0
                candidates = []
                for val in range(last_val + 1, min(last_val + 50, target_scope) + 1):
                    if count_valid_candidates(r, idx, val) > 0:
                        count += 1
                        candidates.append(val)
                        if count > 20:  # cap for efficiency
                            break
                
                if count < best_count:
                    best_count = count
                    best_pos = (r, idx)
                    best_candidates = candidates
                
                if count == 0:
                    return False  # dead end
            
            if best_count == 0:
                return False
            
            # Try candidates in random order (or sorted for determinism)
            random.shuffle(best_candidates)
            
            for val in best_candidates:
                if time.time() - start > time_limit:
                    return None  # timeout
                
                r, idx = best_pos
                # Apply the choice
                rows[r].append(val)
                new_diffs = set()
                for a in rows[r][:-1]:  # exclude the just-added val
                    d = val - a
                    new_diffs.add(d)
                row_diffs[r] |= new_diffs
                used_diffs |= new_diffs
                
                # Recurse
                result = try_mrv_step()
                if result is True:
                    return True
                elif result is None:
                    # Undo and propagate timeout
                    rows[r].pop()
                    row_diffs[r] -= new_diffs
                    used_diffs -= new_diffs
                    return None
                else:
                    # Undo and try next candidate
                    rows[r].pop()
                    row_diffs[r] -= new_diffs
                    used_diffs -= new_diffs
            
            return False
        
        # Reset state and try
        rows = [[0] for _ in range(n_rows)]
        row_diffs = [set() for _ in range(n_rows)]
        used_diffs = set()
        
        result = try_mrv_step()
        
        if result is True:
            return rows
        return None

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Multi-restart strategy
    num_restarts = 12
    for i in range(num_restarts):
        if time_left() < 25:
            break
        
        new_rows = greedy(seed=i+1)
        if new_rows:
            new_scope = scope(new_rows)
            if new_scope < best_scope:
                best_rows = new_rows
                best_scope = new_scope
                if time_left() > 15:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # ILS on the best found so far
    while time_left() > 35:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        if time_left() < 30:
            break
            
        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Phase 3: Targeted Backtracking with MRV to try to beat best_scope
    if time_left() > 15:
        # Try to improve by 1, 2, 3
        for delta in [1, 2, 3]:
            if time_left() < 10:
                break
            target = best_scope - delta
            if target < n_rows * (k * (k + 1) // 2):  # counting bound
                break
            bt_rows = backtrack_mrv(target, time_limit=min(12, time_left() - 5))
            if bt_rows:
                best_rows = bt_rows
                best_scope = target
                if time_left() > 8:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)
                break  # found an improvement, stop trying larger deltas

    # Final polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
