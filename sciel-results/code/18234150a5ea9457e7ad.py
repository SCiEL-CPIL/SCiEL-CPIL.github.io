"""Difference triangle set (4,7) - simulated annealing with difference-pressure scoring."""
import time
import random
import math

def construct(n_rows, k):
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)

    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s
    polish_start = start_time + 128.0  # last ~17s for polish

    def time_up():
        return time.time() > deadline

    def in_polish():
        return time.time() > polish_start

    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1

    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
        return True

    rng = random.Random(12345)

    # Build a single row with difference-pressure scoring.
    # forbidden: set of globally used diffs (excluding this row's own).
    # max_limit: upper bound on the row's max entry.
    # pressure: weight on max-new-diff penalty (higher = prefer compact rows).
    def build_row(k, forbidden, max_limit, pressure=1.0, bias=0.0):
        row = [0]
        local_used = set()
        for step in range(k):
            last = row[-1]
            remaining = k - (step + 1)
            max_x = max_limit - remaining
            if last + 1 > max_x:
                return None
            best_score = None
            best_x = None
            best_diffs = None
            # Scan candidates; use a window to bound cost
            scan_limit = min(max_x, last + 200)
            for x in range(last + 1, scan_limit + 1):
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in local_used or d in forbidden:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                # Score: prefer small x and small max-new-diff
                max_new = max(new_diffs)
                score = x + pressure * max_new
                if bias > 0 and rng.random() < bias:
                    # stochastic: accept with some randomness
                    if best_score is None or rng.random() < 0.5:
                        best_score = score
                        best_x = x
                        best_diffs = new_diffs
                else:
                    if best_score is None or score < best_score:
                        best_score = score
                        best_x = x
                        best_diffs = new_diffs
            if best_x is None:
                return None
            row.append(best_x)
            local_used.update(best_diffs)
        return row

    # Greedy multi-start to get a valid baseline
    def greedy_construct(seed):
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = None
            for attempt in range(50):
                row = build_row(k, used, 2000, pressure=0.5, bias=0.3)
                if row is not None:
                    break
            if row is None:
                return None
            rows.append(row)
            used.update(get_diffs(row))
        return rows, used

    best_rows = None
    best_max = float('inf')
    for seed in range(15):
        if time_up():
            break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        return []

    if best_max == lb:
        return best_rows

    current_rows = [r[:] for r in best_rows]
    current_max = best_max

    # Annealing parameters
    t0 = 1.0
    tmin = 0.01
    total_steps_estimate = 20000
    cool = (tmin / t0) ** (1.0 / max(1, total_steps_estimate))
    temperature = t0

    no_improve = 0
    max_no_improve = 300
    step = 0

    def rebuild_single(i, max_limit, pressure, bias):
        old = current_rows[i]
        old_diffs = get_diffs(old)
        forbidden = set()
        for j, row in enumerate(current_rows):
            if j != i:
                forbidden.update(get_diffs(row))
        new_row = build_row(k, forbidden, max_limit, pressure=pressure, bias=bias)
        if new_row is None:
            return False
        new_diffs = get_diffs(new_row)
        # check no overlap with forbidden (should be guaranteed by builder, but verify)
        if new_diffs & forbidden:
            return False
        current_rows[i] = new_row
        return True

    def rebuild_multi(indices, max_limit, pressure, bias):
        # Rebuild multiple rows simultaneously
        n = len(indices)
        old_rows = [current_rows[i] for i in indices]
        old_diffs_list = [get_diffs(r) for r in old_rows]
        # forbidden = diffs from all other rows
        forbidden = set()
        other_indices = set(range(n_rows)) - set(indices)
        for j in other_indices:
            forbidden.update(get_diffs(current_rows[j]))
        # Build new rows one at a time, accumulating their diffs
        new_rows = []
        new_forbidden = set(forbidden)
        for idx in indices:
            row = build_row(k, new_forbidden, max_limit, pressure=pressure, bias=bias)
            if row is None:
                return False
            new_rows.append(row)
            new_forbidden.update(get_diffs(row))
        # Accept
        for pos, idx in enumerate(indices):
            current_rows[idx] = new_rows[pos]
        return True

    while not time_up():
        step += 1
        temperature = max(tmin, temperature * cool)
        polish = in_polish()

        # Decide move type
        if polish:
            # Strict improvement only, tighter limits
            move_type = 'polish'
        else:
            r = rng.random()
            if r < 0.70:
                move_type = 'single'
            elif r < 0.90:
                move_type = 'multi'
            else:
                move_type = 'reshuffle'

        if move_type == 'single':
            i = rng.randint(0, n_rows - 1)
            old_max = get_max(current_rows)
            if polish:
                # Target strictly smaller
                target = current_max - 1
                if max(current_rows[i]) > target and target > k:
                    ok = rebuild_single(i, target, pressure=1.5, bias=0.0)
                else:
                    ok = False
            else:
                # Try to improve or neutral
                if max(current_rows[i]) == current_max:
                    target = current_max - 1
                else:
                    target = current_max
                if target > k:
                    ok = rebuild_single(i, target, pressure=1.0, bias=0.3)
                else:
                    ok = False

            if not ok:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue

            new_max = get_max(current_rows)
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                # Accept with annealing probability if worse (only non-polish)
                if not polish and new_max > old_max:
                    delta = new_max - old_max
                    if rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                        no_improve += 1
                    else:
                        # revert
                        pass
                else:
                    no_improve += 1
                if no_improve > max_no_improve:
                    break

        elif move_type == 'multi':
            # Pick 2 rows to rebuild
            i1 = rng.randint(0, n_rows - 1)
            i2 = rng.randint(0, n_rows - 1)
            while i2 == i1:
                i2 = rng.randint(0, n_rows - 1)
            old_max = get_max(current_rows)
            target = current_max - 1 if not polish else current_max - 1
            if target > k:
                ok = rebuild_multi([i1, i2], target, pressure=1.2, bias=0.2)
            else:
                ok = False
            if not ok:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue
            new_max = get_max(current_rows)
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        elif move_type == 'reshuffle':
            # Neutral reshuffle to escape local minima
            i = rng.randint(0, n_rows - 1)
            old_max = get_max(current_rows)
            ok = rebuild_single(i, current_max, pressure=0.8, bias=0.7)
            if not ok:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue
            new_max = get_max(current_rows)
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

    # Final validation
    if not validate(best_rows, n_rows, k):
        res = greedy_construct(0)
        if res:
            best_rows = res[0]

    return best_rows
