"""Combined program: Greedy + ILS with targeted backtracking and multi-restart."""
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
        random.seed(seed) if seed > 0 else None
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                # Look ahead window
                # Larger window for later rows or if jitter is active
                window = 100 if (seed > 0 and r_idx > 0) else 50
                for trial_x in range(x + 1, x + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        # Jitter: skip valid candidates to escape local minima
                        if seed > 0 and random.random() < 0.1 and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    # Fallback: expand window significantly
                    for trial_x in range(x + 1, x + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    # Should not happen with sufficient window, but safety
                    break
                x = row[-1]
            if len(row) == k + 1:
                rows.append(row)
            else:
                # If we failed to complete a row (unlikely), just return what we have?
                # Better to restart or fail. For robustness, return None or a dummy.
                # But greedy with window 500 should always work for small n,k.
                pass
        
        # Verify we got all rows
        if len(rows) != n_rows:
            # Fallback to strict greedy
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
        # Strategy: Rebuild 1 or 2 random rows with high randomness
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with high jitter
            row, x = [0], 0
            current_row_diffs = set()
            success = False
            # Try a few times to find a valid row, even if larger
            for attempt in range(25):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    # Wider search range for kicks
                    for trial_x in range(x + 1, x + 120):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            # High jitter: skip valid candidates often
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
                # If we failed to rebuild, keep the original row
                # This might make the solution invalid temporarily, but local_polish will fix it
                # Or we revert to a greedy row
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

    # ---------- Targeted Backtracking Search ----------
    def backtrack_search(target_scope, time_limit):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        Returns rows if found, None otherwise.
        """
        start = time.time()
        
        # Pre-compute all possible diffs up to target_scope
        # We need to fill n_rows * C(k+1, 2) distinct diffs.
        # For n_rows=4, k=7, C(8,2)=28. Total 112 diffs.
        # If target_scope < 112, it's impossible by counting bound.
        
        # State:
        # rows: list of lists
        # used_diffs: set of used diffs
        
        rows = []
        used_diffs = set()
        
        # We build rows one by one.
        # Within a row, we build elements one by one.
        
        def backtrack_row(row_idx, current_row, row_diffs):
            if time.time() - start > time_limit:
                return False
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                used_diffs.update(row_diffs)
                
                if row_idx == n_rows - 1:
                    # All rows complete
                    return True
                
                return backtrack_row(row_idx + 1, [0], set())
            
            # Determine next element
            # It must be > current_row[-1] and <= target_scope
            last_val = current_row[-1]
            
            # Candidate values
            # To make it randomized, we can shuffle the range or pick randomly from valids
            # But we need to check validity.
            
            # Optimization: Only try values that don't immediately conflict?
            # For backtracking, we try candidates.
            
            # Range of candidates: last_val + 1 to target_scope
            # To speed up, we can limit the search space slightly or use heuristics.
            # Let's try a subset of candidates first, then expand if stuck.
            # For simplicity in this implementation, we iterate but with a time check.
            
            # To avoid deep recursion overhead and allow randomization, we can use a stack or recursion.
            # Recursion is fine for depth k+1 (8).
            
            # Heuristic: Try smaller values first? Or random?
            # Random helps escape traps.
            # Let's try a random subset of candidates first.
            
            candidates = list(range(last_val + 1, target_scope + 1))
            random.shuffle(candidates)
            
            # Limit candidates to avoid huge branching factor if target_scope is large
            # But target_scope should be close to best_scope, so it's manageable.
            
            for val in candidates:
                if time.time() - start > time_limit:
                    return False
                
                # Check diffs
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = val - a
                    if d in used_diffs or d in row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Recurse
                    if backtrack_row(row_idx, current_row + [val], row_diffs | new_diffs):
                        return True
                    # Backtrack
                    # (No explicit cleanup needed as we use immutable structures in recursion mostly, 
                    # but current_row is a list. We pass a new list current_row + [val])
            
            return False

        # Start backtracking
        # We need to be careful with the 'rows' and 'used_diffs' being mutated.
        # Let's use a cleaner recursive approach with local state or explicit stack.
        
        def solve_bt():
            rows_local = []
            used_local = set()
            
            def rec(row_idx, curr_row, curr_diffs):
                if time.time() - start > time_limit:
                    return False
                
                if len(curr_row) == k + 1:
                    rows_local.append(curr_row)
                    used_local.update(curr_diffs)
                    
                    if row_idx == n_rows - 1:
                        return True
                    
                    return rec(row_idx + 1, [0], set())
                
                last_val = curr_row[-1]
                candidates = list(range(last_val + 1, target_scope + 1))
                random.shuffle(candidates)
                
                for val in candidates:
                    if time.time() - start > time_limit:
                        return False
                    
                    new_diffs = set()
                    valid = True
                    for a in curr_row:
                        d = val - a
                        if d in used_local or d in curr_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        if rec(row_idx, curr_row + [val], curr_diffs | new_diffs):
                            return True
                
                return False
            
            if rec(0, [0], set()):
                return rows_local
            return None

        return solve_bt()

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Multi-restart strategy
    # 1. Try to find better solutions via randomized greedy restarts
    # 2. Use backtracking to try to beat the current best scope
    
    # Phase 1: Randomized Greedy Restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time_left() < 20:
            break
        
        # Try a randomized greedy start
        new_rows = greedy(seed=i+1)
        if new_rows:
            new_scope = scope(new_rows)
            if new_scope < best_scope:
                best_rows = new_rows
                best_scope = new_scope
                # Polish immediately
                if time_left() > 15:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # Phase 2: ILS on the best found so far
    while time_left() > 30:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 25:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Phase 3: Targeted Backtracking to try to beat best_scope
    # Try to find a solution with scope best_scope - 1, best_scope - 2, etc.
    # This is a very targeted search.
    
    if time_left() > 10:
        # Try to improve by 1
        target = best_scope - 1
        bt_rows = backtrack_search(target, time_limit=10)
        if bt_rows:
            best_rows = bt_rows
            best_scope = target
            # Polish if possible
            if time_left() > 5:
                polished = local_polish(best_rows)
                if scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)
        
        # Try to improve by 2
        if time_left() > 10:
            target = best_scope - 1
            bt_rows = backtrack_search(target, time_limit=10)
            if bt_rows:
                best_rows = bt_rows
                best_scope = target
                if time_left() > 5:
                    polished = local_polish(best_rows)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
