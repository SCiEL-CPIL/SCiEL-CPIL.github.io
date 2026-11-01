"""Difference triangle set solver: ILS with kicks + fast conflict tracking + per-row polish."""
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s kill

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

    # ---------- Fast incremental conflict machinery ----------
    class ConflictTracker:
        def __init__(self, rows):
            self.conf = {}
            self.n_conf = 0
            for row in rows:
                for d in row_diffs_list(row):
                    self._add(d, 1)

        def _add(self, d, delta):
            old = self.conf.get(d, 0)
            new = old + delta
            if old == 1 and new == 2:
                self.n_conf += 1
            elif old == 2 and new == 1:
                self.n_conf -= 1
            if new == 0:
                self.conf.pop(d, None)
            else:
                self.conf[d] = new

        def remove_row(self, row):
            for d in row_diffs_list(row):
                self._add(d, -1)

        def add_row(self, row):
            for d in row_diffs_list(row):
                self._add(d, 1)

        def valid(self):
            return self.n_conf == 0

    # ---------- Row builders ----------
    def greedy_row(used_set, jitter=0.0, cap=10**6):
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

    # ---------- Per-row local optimization ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Adaptive jitter: start deterministic, add randomness later
            jitter = 0.0 if attempt < 5 else (0.05 if attempt < 20 else 0.15)
            # Wider search window for later attempts
            window = 100 if attempt < 10 else 200
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + window):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 400:
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
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=90)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    if scope(rows) < old_scope:
                        improved = True
        return rows

    # ---------- Kick: targeted rebuild of worst row + occasional random kick ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        # 70% targeted kick on the row with the largest entry, 30% random
        if random.random() < 0.7:
            worst_idx = max(range(n_rows), key=lambda i: max(rows[i]))
            indices = [worst_idx]
            if random.random() < 0.3:
                other = [i for i in range(n_rows) if i != worst_idx]
                indices.append(random.choice(other))
        else:
            num_kicks = random.choice([1, 2])
            indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            for attempt in range(40):
                jitter = random.uniform(0.2, 0.5)
                row = greedy_row(used_other, jitter=jitter, cap=500)
                if row is not None and max(row) < max(rows[idx]):
                    rows[idx] = row
                    done = True
                    break
            if not done:
                # Try with less jitter
                for attempt in range(20):
                    row = greedy_row(used_other, jitter=0.1, cap=400)
                    if row is not None:
                        rows[idx] = row
                        done = True
                        break
            if not done:
                row = greedy_row(used_other, jitter=0.0, cap=10**6)
                if row is not None:
                    rows[idx] = row
                else:
                    rows[idx] = list(rows[idx])
        return rows

    # ---------- Initialization: multiple-restart greedy ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)

    if time_left() > 40:
        for r in range(12):
            if time_left() < 40:
                break
            cand = greedy_full(jitter=random.uniform(0.05, 0.3))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # Initial polish
    if time_left() > 20:
        polished = local_polish(best_rows, max_passes=10)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # ---------- ILS main loop: polish -> kick -> polish ----------
    cycles = 0
    while time_left() > 30:
        cycles += 1

        # 1. Local search on incumbent
        polished = local_polish(best_rows, max_passes=6)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        if time_left() < 25:
            break

        # 2. Kick
        kicked = kick(best_rows)

        # 3. Local search on kicked solution
        if time_left() < 20:
            break
        polished2 = local_polish(kicked, max_passes=6)
        if scope(polished2) < best_scope:
            best_rows = polished2
            best_scope = scope(polished2)

    # Final high-precision polish: more passes, tighter per-row reduction
    if time_left() > 15:
        polished = local_polish(best_rows, max_passes=12)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
    
    # Extra tight polish on the worst row
    if time_left() > 8:
        worst_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
        used_other = used_set_except(best_rows, worst_idx)
        new_row, new_max = try_reduce_row(best_rows, worst_idx, used_other, max_attempts=150)
        if new_max < max(best_rows[worst_idx]):
            test_rows = [list(r) for r in best_rows]
            test_rows[worst_idx] = new_row
            if is_valid(test_rows):
                best_rows = test_rows
                best_scope = scope(best_rows)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
