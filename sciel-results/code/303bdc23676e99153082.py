"""Difference triangle set solver: ILS with backtracking row builder + conflict tracking + polish."""
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        diffs = []
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.append(row[j] - ri)
        return diffs

    def row_diffs_set(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
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

    # ---------- Backtracking row builder with MRV-like heuristics ----------
    def backtrack_row(used_set, max_val, jitter=0.0):
        """Build a row using backtracking with constraint propagation.
        Selects next element by trying values that minimize remaining conflicts."""
        row = [0]
        row_diffs = set()
        
        def _build(pos, current_max):
            if pos == k + 1:
                return True
            
            best_candidate = None
            best_conflicts = float('inf')
            
            # Try candidates from current_max+1 upward, with some randomization
            start_x = current_max + 1
            end_x = min(max_val, current_max + 50)
            
            candidates = list(range(start_x, end_x + 1))
            if jitter > 0:
                random.shuffle(candidates)
            else:
                # MRV heuristic: prefer candidates with fewer new conflicts
                scored = []
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    conflicts = len(new_diffs & used_set) + len(new_diffs & row_diffs)
                    scored.append((conflicts, x))
                scored.sort()
                candidates = [x for _, x in scored]
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if new_diffs & used_set or new_diffs & row_diffs:
                    continue
                if jitter > 0 and random.random() < jitter * 0.5 and x < 300:
                    continue
                    
                row.append(x)
                old_diffs = set(row_diffs)
                row_diffs |= new_diffs
                
                if _build(pos + 1, x):
                    return True
                
                row.pop()
                row_diffs = old_diffs
            
            return False
        
        if _build(1, 0):
            return row
        return None

    def greedy_row(used_set, jitter=0.0, cap=10**6):
        """Fast greedy row builder for initialization."""
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            if x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set) and not (new & row_diffs):
                if jitter > 0 and random.random() < jitter and x < 400:
                    continue
                row.append(x)
                row_diffs |= new
        return row

    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

    def greedy_full(jitter=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, jitter=jitter)
            if row is None:
                return None
            rows.append(row)
            used |= row_diffs_set(row)
        return rows

    def interleaved_greedy(jitter=0.0):
        """Build rows in interleaved fashion."""
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(k):
            order = sorted(range(n_rows), key=lambda i: rows[i][-1] + (random.random() * 0.5 if jitter > 0 else 0))
            for i in order:
                row = rows[i]
                x = row[-1] + 1
                found = False
                while x < 10**6:
                    new = {x - a for a in row}
                    row_diffs = set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
                    if not (new & used) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter and x < 400:
                            x += 1
                            continue
                        row.append(x)
                        for d in new:
                            used.add(d)
                        found = True
                        break
                    x += 1
                if not found:
                    return None
        return rows

    # ---------- Per-row local optimization with backtracking ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100, max_val=None):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        
        if max_val is None:
            max_val = current_max - 1
        
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            
            jitter = 0.0 if attempt < 15 else (0.05 + 0.01 * (attempt - 15))
            row = backtrack_row(used_other, max_val, jitter=jitter)
            
            if row is not None and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
                    max_val = best_max - 1  # Tighten for next attempt
        
        return best_row, best_max

    def local_polish(rows, max_passes=8):
        rows = [list(r) for r in rows]
        improved = True
        passes = 0
        while improved and time_left() > 8 and passes < max_passes:
            improved = False
            passes += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 8:
                    break
                used_other = used_set_except(rows, i)
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=60)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    # ---------- Targeted descent ----------
    def targeted_descent(rows, budget=3.0):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = scope(cur)
        improved = False

        while time.time() - start < budget and time_left() > 15:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = random.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                pos2 = random.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(15):
                    nv = random.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    if is_valid(cur) and scope(cur) < cur_scope:
                        cur_scope = scope(cur)
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
                continue

            for new_val in range(old_val - 1, lower - 1, -1):
                row[pos] = new_val
                if is_valid(cur):
                    new_scope = scope(cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        break
                else:
                    row[pos] = old_val
                    break

            if not improved:
                break

        return cur, cur_scope

    # ---------- Kick: rebuild rows with backtracking + high jitter ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            for attempt in range(20):
                jitter = 0.3 + 0.1 * attempt
                max_val = best_scope + 20
                row = backtrack_row(used_other, max_val, jitter=jitter)
                if row is not None:
                    rows[idx] = row
                    done = True
                    break
            if not done:
                row = greedy_row(used_other, jitter=0.0, cap=10**6)
                if row is not None:
                    rows[idx] = row
        return rows

    # ---------- Initialization ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)

    if time_left() > 30:
        for r in range(6):
            if time_left() < 30:
                break
            if r % 2 == 0:
                cand = greedy_full(jitter=random.uniform(0.05, 0.2))
            else:
                cand = interleaved_greedy(jitter=random.uniform(0.05, 0.2))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Initial targeted descent
    if time_left() > 15:
        descended, desc_scope = targeted_descent(best_rows, budget=3.0)
        if desc_scope < best_scope:
            best_rows = descended
            best_scope = desc_scope

    # ---------- ILS main loop ----------
    cycles = 0
    while time_left() > 25:
        cycles += 1

        # Local search on incumbent
        polished = local_polish(best_rows, max_passes=5)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # Targeted descent
        if time_left() > 20:
            descended, desc_scope = targeted_descent(best_rows, budget=2.0)
            if desc_scope < best_scope:
                best_rows = descended
                best_scope = desc_scope

        if time_left() < 20:
            break

        # Periodic full restart
        if cycles % 4 == 0 and time_left() > 40:
            restart_jitter = random.uniform(0.1, 0.3)
            if cycles % 8 == 0:
                cand = interleaved_greedy(jitter=restart_jitter)
            else:
                cand = greedy_full(jitter=restart_jitter)
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s
                    polished = local_polish(best_rows, max_passes=5)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)
                    if time_left() > 25:
                        descended, desc_scope = targeted_descent(best_rows, budget=2.0)
                        if desc_scope < best_scope:
                            best_rows = descended
                            best_scope = desc_scope
                    continue

        # Kick
        kicked = kick(best_rows)

        # Local search on kicked solution
        if time_left() < 15:
            break
        polished2 = local_polish(kicked, max_passes=5)
        if scope(polished2) < best_scope:
            best_rows = polished2
            best_scope = scope(polished2)

        # Targeted descent on kicked solution
        if time_left() > 20:
            descended2, desc_scope2 = targeted_descent(best_rows, budget=2.0)
            if desc_scope2 < best_scope:
                best_rows = descended2
                best_scope = desc_scope2

    # Final polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final targeted descent
    if time_left() > 8:
        descended, desc_scope = targeted_descent(best_rows, budget=3.0)
        if desc_scope < best_scope:
            best_rows = descended
            best_scope = desc_scope

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
