"""Difference triangle set (4,7): bitmask-accelerated search with constraint propagation."""
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
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

    # ---------- Greedy baseline ----------
    def greedy(seed=0):
        if seed > 0:
            random.seed(seed)
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            row_diffs = set()
            while len(row) < k + 1:
                found = False
                window = 150 if (seed > 0 and r_idx > 0) else 80
                for trial_x in range(row[-1] + 1, row[-1] + window):
                    new = {trial_x - a for a in row}
                    if not (new & used) and not (new & row_diffs):
                        if seed > 0 and random.random() < 0.08 and trial_x < 200:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        used |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(row[-1] + 1, row[-1] + 500):
                        new = {trial_x - a for a in row}
                        if not (new & used) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            used |= new
                            found = True
                            break
                if not found:
                    break
            if len(row) == k + 1:
                rows.append(row)
        if len(rows) != n_rows:
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                row_diffs = set()
                while len(row) < k + 1:
                    row[-1] += 1
                    new = {row[-1] - a for a in row[:-1]}
                    if not (new & used) and not (new & row_diffs):
                        row.append(row[-1])
                        row_diffs |= new
                        used |= new
                rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Local optimization ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row = [0]
            row_diffs = set()
            success = True
            jitter = 0.08 + 0.04 * (attempt / max_attempts)
            while len(row) < k + 1:
                found = False
                window = 60 + 30 * (attempt / max_attempts)
                for trial_x in range(row[-1] + 1, int(row[-1] + window)):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        if random.random() < jitter and trial_x < 250:
                            continue
                        row.append(trial_x)
                        row_diffs |= new
                        found = True
                        break
                if not found:
                    for trial_x in range(row[-1] + 1, row[-1] + 200):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs):
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                if not found:
                    success = False
                    break
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
        pass_count = 0
        max_passes = 8
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
                        for d in row_diffs_list(rows[j]):
                            used_other.add(d)
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=120)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    for d in row_diffs_list(rows[j]):
                        used_other.add(d)
            row = [0]
            row_diffs = set()
            success = False
            for attempt in range(20):
                row = [0]
                row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    for trial_x in range(row[-1] + 1, row[-1] + 100):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & row_diffs):
                            if random.random() < 0.4 and trial_x < 300:
                                continue
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            if not success:
                greedy_row = [0]
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    greedy_row[-1] += 1
                    new = {greedy_row[-1] - a for a in greedy_row[:-1]}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(greedy_row[-1])
                        row_diffs |= new
                rows[idx] = greedy_row
        return rows

    # ---------- Backtracking with forward checking ----------
    def backtrack_search(target_scope, time_limit):
        start = time.time()
        rows_local = []
        used_local = set()
        nodes_visited = 0
        max_nodes = 800000

        def rec(row_idx, curr_row, curr_diffs):
            nonlocal nodes_visited
            if time.time() - start > time_limit or nodes_visited > max_nodes:
                return False

            if len(curr_row) == k + 1:
                rows_local.append(list(curr_row))
                used_local.update(curr_diffs)
                if row_idx == n_rows - 1:
                    return True
                return rec(row_idx + 1, [0], set())

            last_val = curr_row[-1]
            candidates = list(range(last_val + 1, target_scope + 1))

            def conflict_score(val):
                new_diffs = {val - a for a in curr_row}
                conflicts = len(new_diffs & used_local) + len(new_diffs & curr_diffs)
                return (conflicts, val)

            if random.random() < 0.25:
                random.shuffle(candidates)
            else:
                candidates.sort(key=conflict_score)

            max_candidates = min(len(candidates), 120)
            candidates = candidates[:max_candidates]

            for val in candidates:
                if time.time() - start > time_limit or nodes_visited > max_nodes:
                    return False
                nodes_visited += 1
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

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Multi-restart
    for i in range(12):
        if time_left() < 25:
            break
        new_rows = greedy(seed=i + 1)
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

    # ILS
    while time_left() > 35:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        if time_left() < 28:
            break
        kicked_rows = kick(best_rows)
        polished_kicked = local_polish(kicked_rows)
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Targeted backtracking
    for improvement in range(1, 4):
        if time_left() < 15 - improvement * 3:
            break
        target = best_scope - improvement
        bt_rows = backtrack_search(target, time_limit=max(5, 12 - improvement * 3))
        if bt_rows:
            best_rows = bt_rows
            best_scope = target
            if time_left() > 5:
                polished = local_polish(best_rows)
                if scope(polished) < best_scope:
                    best_rows = polished
                    best_scope = scope(polished)

    # Final high-precision polish
    if time_left() > 8:
        best_rows = [list(r) for r in best_rows]
        for _ in range(4):
            if time_left() < 3:
                break
            improved = False
            order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
            for i in order:
                if time_left() < 2:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        for d in row_diffs_list(best_rows[j]):
                            used_other.add(d)
                new_row, new_max = try_reduce_row(best_rows, i, used_other, max_attempts=250)
                if new_max < max(best_rows[i]):
                    best_rows[i] = new_row
                    improved = True
            if not improved:
                break
        if scope(best_rows) < best_scope:
            best_scope = scope(best_rows)

    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
