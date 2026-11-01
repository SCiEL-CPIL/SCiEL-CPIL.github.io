"""Difference triangle set solver: ILS with kicks + fast conflict tracking + per-row polish + interleaved greedy."""
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
    def greedy_row(used_set, jitter=0.0, cap=10**6, lookahead=0):
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            if x > cap:
                return None
            new = {x - a for a in row}
            if (new & used_set) or (new & row_diffs):
                continue
            if jitter > 0 and random.random() < jitter and x < 400:
                continue
            # Optional lookahead: check if adding this leads to dead-end soon
            if lookahead > 0 and len(row) < k + 1 - lookahead:
                test_row = row + [x]
                test_diffs = row_diffs | new
                dead_end = False
                for probe_x in range(x + 1, x + lookahead * 10 + 1):
                    probe_new = {probe_x - a for a in test_row}
                    if (probe_new & used_set) or (probe_new & test_diffs):
                        continue
                    # Found a viable next step
                    dead_end = False
                    break
                else:
                    dead_end = True
                if dead_end:
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
            # Order: prefer rows with smaller last element, with jitter
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
    def try_reduce_row(rows, row_idx, used_other, max_attempts=100, target_max=None):
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        if target_max is not None:
            best_max = target_max
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Adaptive jitter: more randomness in later attempts
            jitter = 0.08 + 0.04 * (attempt / max_attempts) if attempt > 5 else 0.0
            while len(row) < k + 1:
                found = False
                # Try a wider window with better spacing
                search_window = 200 if attempt < 20 else 100
                for trial_x in range(x + 1, x + search_window):
                    new = {trial_x - a for a in row}
                    if (new & used_other) or (new & current_row_diffs):
                        continue
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

    def local_polish(rows, max_passes=8, aggressive=False):
        rows = [list(r) for r in rows]
        improved = True
        passes = 0
        while improved and time_left() > 8 and passes < max_passes:
            improved = False
            passes += 1
            # Process rows in descending order of max value
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 8:
                    break
                used_other = used_set_except(rows, i)
                current_max = max(rows[i])
                # Target slightly below current max for aggressive mode
                target = current_max - 1 if aggressive else None
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=120, target_max=target)
                if new_max < current_max:
                    old_scope = scope(rows)
                    rows[i] = new_row
                    if scope(rows) < old_scope:
                        improved = True
        return rows

    # ---------- Kick: rebuild 1-2 rows with high randomness ----------
    def kick(rows, preserve_structure=False):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        # Prefer kicking rows with larger max values
        if preserve_structure:
            indices = sorted(range(n_rows), key=lambda i: -max(rows[i]))[:num_kicks]
        else:
            indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            # Try multiple jitter levels
            for jitter_level in [0.3, 0.5, 0.7]:
                for attempt in range(20):
                    row = greedy_row(used_other, jitter=jitter_level, cap=500)
                    if row is not None:
                        rows[idx] = row
                        done = True
                        break
                if done:
                    break
            if not done:
                # Fallback: greedy without jitter
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
        # Try diverse jitter levels and strategies
        jitter_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
        for r in range(10):
            if time_left() < 30:
                break
            jitter = jitter_levels[r % len(jitter_levels)]
            # Alternate between sequential and interleaved greedy
            if r % 2 == 0:
                cand = greedy_full(jitter=jitter)
            else:
                cand = interleaved_greedy(jitter=jitter)
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # Initial polish - be more aggressive
    if time_left() > 15:
        polished = local_polish(best_rows, aggressive=True)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # ---------- ILS main loop: polish -> kick -> polish, with periodic full restarts ----------
    cycles = 0
    no_improve_count = 0
    while time_left() > 25:
        cycles += 1
        prev_scope = best_scope

        # 1. Local search on incumbent - alternate between normal and aggressive
        polished = local_polish(best_rows, max_passes=6, aggressive=(cycles % 3 == 0))
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if time_left() < 20:
            break

        # 2. Periodic full restart (every 4 cycles or if stuck)
        if (cycles % 4 == 0 or no_improve_count >= 3) and time_left() > 40:
            restart_jitter = random.uniform(0.08, 0.35)
            if cycles % 8 == 0:
                cand = interleaved_greedy(jitter=restart_jitter)
            else:
                cand = greedy_full(jitter=restart_jitter)
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s
                    no_improve_count = 0
                    polished = local_polish(best_rows, max_passes=6, aggressive=True)
                    if scope(polished) < best_scope:
                        best_rows = polished
                        best_scope = scope(polished)
                    continue

        # 3. Kick - preserve structure when stuck
        preserve = no_improve_count >= 2
        kicked = kick(best_rows, preserve_structure=preserve)

        # 4. Local search on kicked solution
        if time_left() < 15:
            break
        polished2 = local_polish(kicked, max_passes=6, aggressive=(no_improve_count >= 2))
        if scope(polished2) < best_scope:
            best_rows = polished2
            best_scope = scope(polished2)
            no_improve_count = 0

    # Final polish - most aggressive
    if time_left() > 10:
        polished = local_polish(best_rows, max_passes=10, aggressive=True)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy_full(jitter=0.0)

    return best_rows
