"""Combined program: Greedy + ILS with Backtracking Row Reconstruction."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin under 150s kill

    def time_left():
        return deadline - time.time()

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
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy(jitter=0.0):
        """Greedy construction with optional jitter."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
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
                        new = {x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(x)
                            row_diffs |= new
                            used |= new
                            break
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # Initial restarts to find a good starting point
    if time_left() > 30:
        for _ in range(8):
            if time_left() < 30:
                break
            cand = greedy(jitter=random.uniform(0.05, 0.25))
            s = scope(cand)
            if s < best_scope:
                best_rows = cand
                best_scope = s

    # ---------- Backtracking row reconstruction ----------
    def backtrack_row(target_max, used_other, time_limit, seed=None):
        """
        Attempts to find a valid row with max value <= target_max.
        Returns the row if found, None otherwise.
        """
        if seed is not None:
            random.seed(seed)
        start = time.time()
        row = [0] * (k + 1)
        
        def solve(pos):
            if time.time() > start + time_limit:
                return False
            
            if pos == k + 1:
                return True
            
            prev_val = row[pos - 1] if pos > 0 else -1
            lower = prev_val + 1
            upper = target_max
            
            if lower > upper:
                return False
            
            # Collect valid candidates
            candidates = []
            # Scan a window first for efficiency
            scan_limit = min(upper, lower + 50)
            
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - row[i]
                    if d in used_other:
                        valid = False
                        break
                    # Check against existing diffs in this row
                    # We can maintain a set of internal diffs for efficiency, but for simplicity
                    # we check against the current row's diffs directly.
                    # Optimization: Precompute internal diffs? No, let's just check.
                    
                if valid:
                    # Check against existing diffs in this row
                    new_diffs = {val - row[i] for i in range(pos)}
                    # Existing diffs are {row[j]-row[i] for i<j<pos}
                    # We can check if any d in new_diffs is in existing diffs
                    existing_diffs = set()
                    for i in range(pos):
                        for j in range(i+1, pos):
                            existing_diffs.add(row[j] - row[i])
                    
                    if not (new_diffs & existing_diffs):
                        candidates.append(val)
                        
            if not candidates and scan_limit < upper:
                # Scan the rest, but limit the number of candidates we keep to avoid memory/time blowup
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - row[i]
                        if d in used_other:
                            valid = False
                            break
                    if valid:
                        new_diffs = {val - row[i] for i in range(pos)}
                        existing_diffs = set()
                        for i in range(pos):
                            for j in range(i+1, pos):
                                existing_diffs.add(row[j] - row[i])
                        if not (new_diffs & existing_diffs):
                            candidates.append(val)
                            if len(candidates) >= 20: # Cap candidates
                                break
            
            if not candidates:
                return False
            
            # Sort candidates: prefer smaller values to keep row tight
            candidates.sort()
            # Randomize a bit: shuffle the first few to escape local minima
            if len(candidates) > 3:
                random.shuffle(candidates[:3])
            
            for val in candidates:
                row[pos] = val
                if solve(pos + 1):
                    return True
                row[pos] = 0
            
            return False

        if solve(1):
            return row[:]
        return None

    def try_reduce_row(rows, row_idx, used_other, max_attempts=10):
        """Try to find a row at row_idx with smaller max entry using backtracking."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 5:
                break

            # Decrease the search space gradually
            target_max = current_max - 1 - (attempt // 5)
            if target_max < k:  # Can't have less than k
                break

            # Try backtracking with a time limit
            time_limit = 1.0
            row = backtrack_row(target_max, used_other, time_limit)
            if row is not None:
                # Verify it's valid in context
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    if max(row) < best_max:
                        best_row = list(row)
                        best_max = max(row)

        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 8
        pass_count = 0
        while improved and time_left() > 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 10:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=5)
                if new_max < max(rows[i]):
                    rows[i] = new_row
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

            # Try to rebuild with a larger max_span to allow exploration
            # We don't have a "build_row_smart" here, so we use greedy with jitter
            success = False
            for attempt in range(10):
                # Use a larger max_span for kicks to allow exploration
                # Greedy doesn't have a max_span param, so we just run it
                # To make it "kicky", we can introduce more jitter
                jitter = random.uniform(0.1, 0.4)
                # We need a way to build a single row given used_other
                # Let's implement a single-row greedy
                row, x = [0], 0
                row_diffs = set()
                valid = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 100):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs):
                            if jitter > 0 and random.random() < jitter:
                                continue
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                    x = row[-1]
                
                if valid:
                    rows[idx] = row
                    success = True
                    break

            if not success:
                # Fallback to standard greedy
                row, x = [0], 0
                row_diffs = set()
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        row.append(x)
                        row_diffs |= new
                rows[idx] = row

        return rows

    # Initial polish
    if time_left() > 20:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
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

    # Final polish
    if time_left() > 5:
        while time_left() > 3:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)
            else:
                break

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
