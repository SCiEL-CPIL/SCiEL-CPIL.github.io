"""Combined program: Greedy + Fixed Backtracking + ILS Polish + Metropolis Refinement."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
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

    # ---------- Greedy baseline ----------
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
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            success = True
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 100):
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
                    while True:
                        x += 1
                        if x > cap:
                            success = False
                            break
                        new = {x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(x)
                            row_diffs |= new
                            break
                    if not success:
                        break
            if not success:
                return None
            rows.append(row)
        return rows

    # Initialize with multiple restarts
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)
    
    if time_left() > 20:
        for _ in range(12):
            if time_left() < 20:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.3))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # ---------- FIXED Backtracking Search with Target Scope ----------
    def backtrack_search(target_scope, time_limit, seed=0):
        """
        FIXED: Now tracks local differences within the current row properly.
        """
        random.seed(seed)
        start = time.time()
        
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        used_global = set()
        used_local = set()  # Differences within current row being built
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                # Row complete, clear local diffs and move to next row
                used_local.clear()
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1] if pos > 0 else -1
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            # Collect valid candidates
            candidates = []
            scan_limit = min(upper, lower + 80)
            
            for val in range(lower, scan_limit + 1):
                valid = True
                # Check against global used
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_global:
                        valid = False
                        break
                if valid:
                    # Check against local used (differences within this row so far)
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_local:
                            valid = False
                            break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 20:
                        break
            
            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_global:
                            valid = False
                            break
                    if valid:
                        for i in range(pos):
                            d = val - rows[row_idx][i]
                            if d in used_local:
                                valid = False
                                break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 20:
                            break
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_global.add(d)
                    used_local.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                # Backtrack
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_global.remove(d)
                    used_local.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 1: Iterative Backtracking Descent
    current_target = best_scope - 1
    seed_counter = 0
    consecutive_failures = 0
    
    while time_left() > 25 and current_target > 100:
        if consecutive_failures > 3:
            break
            
        time_budget = 2.0
        if time_left() > 50 and current_target == best_scope - 1:
            time_budget = 8.0

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
                found = backtrack_search(current_target, time_budget, seed=seed_counter)
                seed_counter += 1
                if found:
                    best_rows = found
                    best_scope = current_target
                    consecutive_failures = 0
                    current_target -= 1
            else:
                current_target -= 1

    # ---------- Per-row local optimization ----------
    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

    def try_reduce_row(rows, row_idx, used_other, max_attempts=200):
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
            jitter = 0.1 if attempt > 20 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 100):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 250:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            row.append(x)
                            current_row_diffs |= new
                            break
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
                used_other = used_set_except(rows, i)
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=100)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = used_set_except(rows, idx)
            
            row, x = [0], 0
            current_row_diffs = set()
            success = False
            for attempt in range(30):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            if random.random() < 0.4 and trial_x < 350:
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

    # ---------- Metropolis Refinement ----------
    def metropolis_refine(rows, steps=5000, t_start=50.0, t_end=1.0):
        rows = [list(r) for r in rows]
        cur_scope = scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = cur_scope
        
        def is_valid_fast(rows):
            seen = set()
            for row in rows:
                for i in range(len(row)):
                    ri = row[i]
                    for j in range(i+1, len(row)):
                        d = row[j] - ri
                        if d in seen:
                            return False
                        seen.add(d)
            return True

        t = t_start
        for step in range(steps):
            if time_left() < 5:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k)
            
            old_val = rows[r_idx][p_idx]
            
            prev_val = rows[r_idx][p_idx - 1]
            next_val = rows[r_idx][p_idx + 1] if p_idx < k else best_scope + 100
            
            lower = prev_val + 1
            upper = min(next_val - 1, best_scope + 10)
            
            if lower > upper:
                continue
            
            if random.random() < 0.7:
                new_val = random.randint(lower, old_val)
            else:
                new_val = random.randint(lower, upper)
            
            if new_val == old_val:
                continue
            
            rows[r_idx][p_idx] = new_val
            
            if not is_valid_fast(rows):
                rows[r_idx][p_idx] = old_val
                continue
            
            new_scope = scope(rows)
            
            delta = new_scope - cur_scope
            if delta < 0:
                cur_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                if random.random() < math.exp(-delta / t):
                    cur_scope = new_scope
                else:
                    rows[r_idx][p_idx] = old_val
            
            t = t_start * (t_end / t_start) ** (step / steps)
            
        return best_rows, best_scope

    # Phase 2: Initial Polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 3: Iterated Local Search with Metropolis
    while time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        if time_left() < 12:
            break
            
        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
            
        met_budget = 4.0
        if time_left() > 25:
            met_budget = 8.0
            
        steps = int(met_budget * 2500)
        refined, refined_scope = metropolis_refine(best_rows, steps=steps, t_start=100.0, t_end=1.0)
        
        if refined_scope < best_scope:
            best_rows = refined
            best_scope = refined_scope

    # Final polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            
        if time_left() > 5:
            refined, refined_scope = metropolis_refine(best_rows, steps=2500, t_start=50.0, t_end=0.5)
            if refined_scope < best_scope:
                best_rows = refined
                best_scope = refined_scope

    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
