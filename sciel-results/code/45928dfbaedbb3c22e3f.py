"""Difference triangle set solver: ILS with kicks + fast conflict tracking + per-row polish + interleaved greedy.
Combined with targeted descent and smarter rebuild from inspiration."""
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

    # ---------- Fast incremental conflict machinery ----------
    class ConflictTracker:
        def __init__(self, rows):
            max_d = max(max(r) for r in rows) if rows else 0
            self.conf = [0] * (max_d + 2)
            self.n_conf = 0
            for row in rows:
                for d in row_diffs_list(row):
                    self._add(d, 1)

        def _ensure(self, d):
            if d >= len(self.conf):
                self.conf.extend([0] * (d + 1 - len(self.conf)))

        def _add(self, d, delta):
            self._ensure(d)
            old = self.conf[d]
            new = old + delta
            if old == 1 and new == 2:
                self.n_conf += 1
            elif old == 2 and new == 1:
                self.n_conf -= 1
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

    def interleaved_greedy(jitter=0.0):
        """Build rows in interleaved fashion: add one element to each row per round."""
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
                    if not (new & used) and not (new & set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))):
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
            jitter = 0.12 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 150):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 300:
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

    # ---------- Targeted descent: try to reduce the maximum entry by adjusting individual positions ----------
    def targeted_descent(rows, budget=3.0):
        """Try to reduce scope by shrinking the largest entry in max rows."""
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
                # Try interior positions
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

            # Try all values from old_val-1 down to lower
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

    # ---------- Kick: rebuild 1-2 rows with high randomness ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            for attempt in range(30):
                row = greedy_row(used_other, jitter=0.4, cap=600)
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

    # ---------- Initialization: multiple-restart greedy (both strategies) ----------
    best_rows = greedy_full(jitter=0.0)
    best_scope = scope(best_rows)

    if time_left() > 30:
        for r in range(8):
            if time_left() < 30:
                break
            # Alternate between sequential and interleaved greedy
            if r % 2 == 0:
                cand = greedy_full(jitter=random.uniform(0.05, 0.25))
            else:
                cand = interleaved_greedy(jitter=random.uniform(0.05, 0.25))
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

    # ---------- ILS main loop: polish -> targeted -> kick -> polish, with periodic full restarts ----------
    cycles = 0
    while time_left() > 25:
        cycles += 1

        # 1. Local search on incumbent
        polished = local_polish(best_rows, max_passes=5)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 1b. Targeted descent
        if time_left() > 20:
            descended, desc_scope = targeted_descent(best_rows, budget=2.0)
            if desc_scope < best_scope:
                best_rows = descended
                best_scope = desc_scope

        if time_left() < 20:
            break

        # 2. Periodic full restart (every 5 cycles) to escape deep local optima
        if cycles % 5 == 0 and time_left() > 40:
            restart_jitter = random.uniform(0.1, 0.3)
            if cycles % 10 == 0:
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
                    # Also try targeted descent on the new best
                    if time_left() > 25:
                        descended, desc_scope = targeted_descent(best_rows, budget=2.0)
                        if desc_scope < best_scope:
                            best_rows = descended
                            best_scope = desc_scope
                    continue

        # 3. Kick
        kicked = kick(best_rows)

        # 4. Local search on kicked solution
        if time_left() < 15:
            break
        polished2 = local_polish(kicked, max_passes=5)
        if scope(polished2) < best_scope:
            best_rows = polished2
            best_scope = scope(polished2)

        # 4b. Targeted descent on kicked solution
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
