"""Combined program: Greedy + Backtracking Descent + ILS Polish."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin under 150s kill

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
        """All positive differences in a row (as a set)."""
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

    def greedy_full(jitter=0.0, cap=10**6):
        """Greedy construction with optional jitter to explore alternatives."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            success = True
            while len(row) < k + 1:
                found = False
                # Look ahead a bit to allow jitter to skip
                for trial_x in range(x + 1, x + 80):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 400:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    # Fallback: standard greedy step
                    while True:
                        x += 1
                        if x > cap:
                            success = False
                            break
                        new = {x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(x)
                            row_diffs |= new
                            used |= new
                            break
                    if not success:
                        break
            if not success:
                return None
            rows.append(row)
        return rows

    # Initialize with multiple restarts of greedy to find a good starting point
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)
    
    if time_left() > 30:
        for _ in range(8):
            if time_left() < 30:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.25))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # ---------- Backtracking Search with Target Scope ----------
    def backtrack_search(target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        Returns rows if found, None otherwise.
        """
        random.seed(seed)
        start = time.time()
        
        # State
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1] if pos > 0 else -1
            # Lower bound is prev_val + 1
            # Upper bound is target_scope
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            # Collect valid candidates
            candidates = []
            # Scan a window first to keep it fast
            scan_limit = min(upper, lower + 60)
            
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 15:
                        break
            
            # If no candidates found in the window, scan the rest up to upper
            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 15:
                            break
            
            if not candidates:
                return False
            
            # Shuffle candidates to randomize the search path
            random.shuffle(candidates)
            
            for val in candidates:
                # Add differences
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                # Backtrack
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 1: Iterative Backtracking Descent
    # Try to reduce the scope by 1 each time using backtracking
    current_target = best_scope - 1
    seed_counter = 0
    consecutive_failures = 0
    
    while time_left() > 30 and current_target > 100: # 100 is a safe lower bound for most small k
        if consecutive_failures > 3:
            break
            
        # Allocate time for this search attempt
        time_budget = 3.0
        
        # If we have a lot of time, we can afford a longer search for the first few targets
        if time_left() > 60 and current_target == best_scope - 1:
            time_budget = 10.0

        found = backtrack_search(current_target, time_budget, seed=seed_counter)
        seed_counter += 1
        
        if found:
            best_rows = found
            best_scope = current_target
            consecutive_failures = 0
            current_target -= 1
        else:
            consecutive_failures += 1
            if consecutive_failures == 1:
                 # Retry with different seed
                 found = backtrack_search(current_target, time_budget, seed=seed_counter)
                 seed_counter += 1
                 if found:
                     best_rows = found
                     best_scope = current_target
                     consecutive_failures = 0
                     current_target -= 1
            else:
                 # Give up on this target, try lower one
                 current_target -= 1

    # ---------- Per-row local optimization (refinement machinery) ----------
    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

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
                    # Fallback
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            row.append(x)
                            current_row_diffs |= new
                            break
                    success = False # If we had to go beyond the lookahead, it's likely not an improvement over greedy
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
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = used_set_except(rows, i)
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
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = used_set_except(rows, idx)
            
            # Rebuild the row with high jitter
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

    # Phase 2: Initial Polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 3: Iterated Local Search
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
