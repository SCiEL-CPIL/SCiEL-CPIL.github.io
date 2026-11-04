"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

_DEADLINE = None
_TIME_LIMIT = 115.0


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    total_diffs = n_rows * k * (k + 1) // 2

    # Counting lower bound
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    rng = random.Random(42)

    # --- Deterministic greedy baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)

    # --- Scope-budgeted construction with repair (STRUCTURAL CHANGE) ---
    def scope_construct(target_scope, rng):
        """
        Try to construct all rows with max entry <= target_scope.
        Uses randomized greedy within the budget, with backtracking repair.
        Returns (rows, success).
        """
        used = set()
        rows = []
        for row_idx in range(n_rows):
            if _time_left() < 3.0:
                return None, False
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find valid candidates within [x+1, target_scope]
                valid = []
                for candidate in range(x + 1, target_scope + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid.append(candidate)
                        if len(valid) >= 25:
                            break

                if not valid:
                    return None, False  # Cannot fit within budget

                # Choose with bias: mostly smallest, sometimes random
                if rng.random() < 0.6:
                    x = valid[0]
                else:
                    # Weighted toward smaller
                    w = [1.0 / (i + 1) for i in range(len(valid))]
                    total_w = sum(w)
                    r = rng.random() * total_w
                    cum = 0.0
                    x = valid[-1]
                    for i in range(len(valid)):
                        cum += w[i]
                        if r <= cum:
                            x = valid[i]
                            break

                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)

        return rows, True

    # --- Repair: given a partially valid set, try to fix conflicts ---
    def repair_rows(rows, target_scope, rng, max_repair_iters=50):
        """
        Given rows that may have conflicts (or just need optimization),
        try to reduce max entry and fix any issues by reassigning elements.
        """
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        for _ in range(max_repair_iters):
            if _time_left() < 3.0:
                break

            # Find the row with the largest entry
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            max_val = row[-1]

            if max_val <= target_scope:
                # Try to reduce further
                pass

            # Try to reduce the last element
            found = False
            for new_val in range(row[-2] + 1, max_val):
                old_diffs = {row[-1] - a for a in row[:-1]}
                new_diffs = {new_val - a for a in row[:-1]}
                if new_diffs & (used - old_diffs):
                    continue
                if new_val <= row[-2]:
                    continue
                used -= old_diffs
                used |= new_diffs
                row[-1] = new_val
                found = True
                break

            if not found:
                # Try to reduce second-to-last
                if len(row) >= 3:
                    pen_val = row[-2]
                    for new_pen in range(row[-3] + 1, pen_val):
                        old_diffs = set()
                        for a in row[:-2]:
                            old_diffs.add(pen_val - a)
                            old_diffs.add(row[-1] - a)
                        old_diffs.add(row[-1] - pen_val)

                        new_pen_diffs = {new_pen - a for a in row[:-2]}
                        if new_pen_diffs & (used - old_diffs):
                            continue
                        if new_pen <= row[-3]:
                            continue

                        best_last = None
                        for new_last in range(new_pen + 1, row[-1] + 1):
                            new_last_diffs = {new_last - a for a in row[:-2]}
                            new_last_diffs.add(new_last - new_pen)
                            all_new = new_pen_diffs | new_last_diffs
                            if all_new & (used - old_diffs):
                                continue
                            if best_last is None or new_last < best_last:
                                best_last = new_last

                        if best_last is not None:
                            used -= old_diffs
                            new_pen_diffs = {new_pen - a for a in row[:-2]}
                            new_last_diffs = {best_last - a for a in row[:-2]}
                            new_last_diffs.add(best_last - new_pen)
                            used |= new_pen_diffs
                            used |= new_last_diffs
                            row[-2] = new_pen
                            row[-1] = best_last
                            found = True
                            break

            if not found:
                break

        return rows

    # --- Full local search (from current program) ---
    def local_search(rows, max_iters=200):
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        improved = False
        for _ in range(max_iters):
            if _time_left() < 2.0:
                break

            row_order = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)
            any_progress = False
            for max_idx in row_order:
                if _time_left() < 2.0:
                    break
                row = rows[max_idx]
                max_val = row[-1]

                found = False
                for new_val in range(row[-2] + 1, max_val):
                    old_diffs = {row[-1] - a for a in row[:-1]}
                    new_diffs = {new_val - a for a in row[:-1]}
                    if new_diffs & (used - old_diffs):
                        continue
                    if new_val <= row[-2]:
                        continue
                    used -= old_diffs
                    used |= new_diffs
                    row[-1] = new_val
                    found = True
                    improved = True
                    any_progress = True
                    break

                if not found and len(row) >= 3:
                    pen_val = row[-2]
                    for new_pen in range(row[-3] + 1, pen_val):
                        old_diffs = set()
                        for a in row[:-2]:
                            old_diffs.add(pen_val - a)
                            old_diffs.add(row[-1] - a)
                        old_diffs.add(row[-1] - pen_val)

                        new_pen_diffs = {new_pen - a for a in row[:-2]}
                        if new_pen_diffs & (used - old_diffs):
                            continue
                        if new_pen <= row[-3]:
                            continue

                        best_last = None
                        for new_last in range(new_pen + 1, row[-1] + 1):
                            new_last_diffs = {new_last - a for a in row[:-2]}
                            new_last_diffs.add(new_last - new_pen)
                            all_new = new_pen_diffs | new_last_diffs
                            if all_new & (used - old_diffs):
                                continue
                            if best_last is None or new_last < best_last:
                                best_last = new_last

                        if best_last is not None:
                            used -= old_diffs
                            new_pen_diffs = {new_pen - a for a in row[:-2]}
                            new_last_diffs = {best_last - a for a in row[:-2]}
                            new_last_diffs.add(best_last - new_pen)
                            used |= new_pen_diffs
                            used |= new_last_diffs
                            row[-2] = new_pen
                            row[-1] = best_last
                            found = True
                            improved = True
                            any_progress = True
                            break

            if not any_progress:
                break

        if improved:
            return rows
        return None

    # --- Validate ---
    def validate(rows):
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
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # --- Main search loop ---
    # Phase 1: Try scope-budgeted construction with decreasing target scopes
    # Start from the greedy max and try to go lower
    current_target = best_max
    for scope_attempt in range(30):
        if _time_left() < 10.0:
            break

        # Try a range of target scopes
        for frac in [1.0, 0.95, 0.9, 0.85, 0.8]:
            target = int(best_max * frac)
            if target < lower_bound:
                continue

            for trial in range(5):
                if _time_left() < 5.0:
                    break
                rows, success = scope_construct(target, rng)
                if not success:
                    continue
                cur_max = max(max(r) for r in rows)
                if cur_max < best_max:
                    # Apply local search to improve further
                    if _time_left() > 5.0:
                        improved = local_search(rows, max_iters=100)
                        if improved:
                            new_max = max(max(r) for r in improved)
                            if new_max < best_max:
                                best_max = new_max
                                best_rows = improved
                    else:
                        best_max = cur_max
                        best_rows = rows
                    break

    # Phase 2: Randomized greedy + local search (from current program)
    for attempt in range(50):
        if _time_left() < 8.0:
            break

        # Randomized greedy
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                window = max(30, 70 - 8 * row_idx)
                valid = []
                for candidate in range(x + 1, x + window + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid.append(candidate)
                        if len(valid) >= 15:
                            break
                if not valid:
                    candidate = x + 1
                    while True:
                        new_diffs = {candidate - a for a in row}
                        if not new_diffs & used:
                            valid.append(candidate)
                            break
                        candidate += 1

                if len(valid) == 1:
                    x = valid[0]
                else:
                    if rng.random() < 0.5:
                        x = valid[0]
                    else:
                        w = [1.0 / (i + 1) for i in range(len(valid))]
                        total_w = sum(w)
                        r = rng.random() * total_w
                        cum = 0.0
                        x = valid[-1]
                        for i in range(len(valid)):
                            cum += w[i]
                            if r <= cum:
                                x = valid[i]
                                break
                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)

        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            if _time_left() > 5.0:
                improved = local_search(best_rows, max_iters=100)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = improved

    # Phase 3: Intensive local search on best
    if _time_left() > 5.0:
        improved = local_search(best_rows, max_iters=400)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    if _time_left() > 3.0:
        improved = local_search(best_rows, max_iters=200)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # Final validation
    if not validate(best_rows):
        best_rows = greedy_construct()

    return best_rows
