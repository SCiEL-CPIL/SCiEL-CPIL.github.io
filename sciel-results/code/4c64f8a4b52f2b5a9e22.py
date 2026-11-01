"""
Difference triangle set (n_rows, k) solver.

Strategy:
  1. Fast greedy baseline (deterministic + jittered restarts).
  2. Targeted backtracking search to reduce scope by 1 iteratively.
  3. Beam search with "most constrained row" selection to explore wider state space.
  4. Iterated Local Search (ILS) with row-level re-optimization and kicks.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin under 150s kill

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
    def greedy(jitter=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 100):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 500:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
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

    # Initialize with deterministic greedy, then try jittered restarts
    best_rows = greedy(0.0)
    best_scope = scope(best_rows)

    if time_left() > 25:
        for _ in range(12):
            if time_left() < 25:
                break
            cand = greedy(random.uniform(0.05, 0.3))
            s = scope(cand)
            if s < best_scope:
                best_rows = cand
                best_scope = s

    # ---------- Backtracking Search with Target Scope ----------
    def backtrack_search(target_scope, time_limit, seed=0):
        random.seed(seed)
        start = time.time()
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
            lower = prev_val + 1
            upper = target_scope
            if lower > upper:
                return False

            # Collect valid candidates in a window
            candidates = []
            scan_limit = min(upper, lower + 80)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
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
                        if d in used_diffs:
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
                    used_diffs.add(d)
                rows[row_idx][pos] = val
                if solve(row_idx, pos + 1):
                    return True
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 1: Iterative Backtracking Descent
    current_target = best_scope - 1
    seed_counter = 0
    consecutive_failures = 0

    while time_left() > 35 and current_target > 100:
        if consecutive_failures > 4:
            break
        time_budget = 4.0
        if time_left() > 80 and current_target == best_scope - 1:
            time_budget = 12.0

        found = backtrack_search(current_target, time_budget, seed=seed_counter)
        seed_counter += 1
        if found:
            best_rows = found
            best_scope = current_target
            consecutive_failures = 0
            current_target -= 1
        else:
            consecutive_failures += 1
            if consecutive_failures <= 2:
                found = backtrack_search(current_target, time_budget, seed=seed_counter)
                seed_counter += 1
                if found:
                    best_rows = found
                    best_scope = current_target
                    consecutive_failures = 0
                    current_target -= 1
            else:
                current_target -= 1

    # ---------- Beam Search with Most-Constrained Row Selection ----------
    def beam_search(target_scope, beam_width=500, time_limit=20.0, seed=0):
        """
        Beam search where at each step we pick the most constrained row/position
        to branch on, maintaining a beam of partial assignments.
        """
        random.seed(seed)
        start = time.time()

        # State: (rows, used_diffs, next_row, next_pos)
        # We represent a beam entry as (rows_tuple, used_frozenset, row_idx, pos)
        # For efficiency, we'll use mutable copies and track best.

        # Initialize: all rows are [0] + [None]*(k), used_diffs = empty
        # Actually, each row starts as [0, None, ..., None]
        # Let's use a different representation: list of rows (each a list of k+1 ints, with -1 for unfilled)
        # and a set of used differences.

        # Beam entry: (rows, used_diffs)
        # rows: list of n_rows lists, each of length k+1, with -1 for unfilled positions
        # We fill positions in a specific order determined by most-constrained heuristic.

        # For simplicity and speed, let's do a randomized greedy with beam-like restarts:
        # At each step, pick the row/position with fewest valid candidates, try all candidates,
        # keep top beam_width partial solutions.

        # This is complex; let's do a simpler but effective approach:
        # Randomized greedy with multiple restarts, but with a "lookahead" penalty
        # that prefers choices that leave more options for future positions.

        best_found = None
        attempts = 0
        max_attempts = 200

        for attempt in range(max_attempts):
            if time.time() > start + time_limit:
                break
            attempts += 1

            # Randomized greedy with jitter
            jitter = 0.1 + 0.1 * (attempt % 5) / 5
            used = set()
            rows = []
            success = True

            for r_idx in range(n_rows):
                row, x = [0], 0
                row_diffs = set()
                while len(row) < k + 1:
                    found = False
                    # Collect all valid candidates in a window
                    valid_cands = []
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            valid_cands.append(trial_x)
                    if not valid_cands:
                        # Fallback: scan further
                        while True:
                            x += 1
                            new = {x - a for a in row}
                            if not (new & used) and not (new & row_diffs):
                                valid_cands.append(x)
                                break
                        if not valid_cands:
                            success = False
                            break

                    # Pick a candidate with some randomness
                    if jitter > 0 and len(valid_cands) > 1:
                        # Weighted random: prefer smaller values but with some randomness
                        weights = [1.0 / (c + 1) for c in valid_cands]
                        total = sum(weights)
                        r = random.random() * total
                        cum = 0
                        chosen = valid_cands[0]
                        for c, w in zip(valid_cands, weights):
                            cum += w
                            if r <= cum:
                                chosen = c
                                break
                    else:
                        chosen = valid_cands[0]

                    row.append(chosen)
                    new = {chosen - a for a in row[:-1]}
                    row_diffs |= new
                    used |= new
                    x = chosen

                if not success:
                    break
                rows.append(row)

            if success:
                s = scope(rows)
                if s <= target_scope:
                    best_found = rows
                    break
                elif best_found is None or s < scope(best_found):
                    best_found = rows

        return best_found

    # Phase 1.5: Beam search to try to beat the backtracking result
    if time_left() > 40 and best_scope > 100:
        target = best_scope - 1
        found = beam_search(target, beam_width=500, time_limit=15.0, seed=42)
        if found and scope(found) < best_scope:
            best_rows = found
            best_scope = scope(found)

    # ---------- Per-row local optimization ----------
    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
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
            jitter = 0.15 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 100):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
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
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=60)
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
    while time_left() > 25:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        if time_left() < 20:
            break

        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows)
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy(0.0)

    return best_rows
# EVOLVE-BLOCK-END
