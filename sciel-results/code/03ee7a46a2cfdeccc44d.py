import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135  # safety margin before 150s wall-clock

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None, randomize=False):
        """Smallest-next-mark row avoiding used_set. Returns None if cap too tight."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline, order=None):
        """Find a row with last element <= cap avoiding used_set. Returns row or None."""
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)  # each remaining step adds at least 1
        if upper < last + 1:
            return None
        if order is not None:
            candidates = [x for x in order if last + 1 <= x <= upper]
        else:
            candidates = range(last + 1, upper + 1)
        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, order)
            if res is not None:
                return res
        return None

    # ---- Phase 1: greedy baseline (always valid) ----
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        rows.append(row)
        used |= row_diffs(row)

    # ---- Phase 2: per-row DFS refinement during construction ----
    # Re-do each row with DFS using the current used set, trying to shrink it.
    for i in range(n_rows):
        if time.time() > deadline - 10:
            break
        cur = rows[i]
        cur_last = cur[-1]
        # Try to find a smaller row for position i
        best = None
        for cap in range(cur_last - 1, max(cur_last - 40, 0), -1):
            if time.time() > deadline - 10:
                break
            used_without = set(used)
            # Remove diffs from rows i..end (they will be rebuilt)
            for j in range(i, n_rows):
                used_without.difference_update(row_diffs(rows[j]))
            row_deadline = min(deadline - 10, time.time() + 2)
            cand = dfs_row([0], used_without, cap, row_deadline)
            if cand is not None:
                best = cand
                break
        if best is not None and best[-1] < cur_last:
            # Rebuild rows from i onward
            for j in range(i, n_rows):
                used.difference_update(row_diffs(rows[j]))
            used |= row_diffs(best)
            rows[i] = best
            for j in range(i + 1, n_rows):
                row = greedy_row(set(used), k)
                rows[j] = row
                used |= row_diffs(row)

    # ---- Phase 3: bounded iterative improvement (replace rows with smaller ones) ----
    # A replaced row frees its old differences, which can let a *smaller* row fit.
    # We only ever shrink the max entry, so the result stays valid and improves.
    order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r in order:
            if time.time() > deadline - 5:
                break
            cur_last = rows[r][-1]
            # Try to fit a strictly smaller row using freed slots.
            best = None
            # Start from cur_last - 1 and go down, but limit the search range
            # to avoid excessive DFS calls. We'll try a few caps.
            for cap in range(cur_last - 1, max(cur_last - 35, 0), -1):
                if time.time() > deadline - 5:
                    break
                used_without = set(used)
                used_without.difference_update(row_diffs(rows[r]))
                row_deadline = min(deadline - 5, time.time() + 2)
                cand = dfs_row([0], used_without, cap, row_deadline)
                if cand is not None:
                    best = cand
                    break
            if best is not None and best[-1] < cur_last:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = best
                used |= row_diffs(best)
                improved = True

    # ---- Phase 4: local search to reduce max entry within a row ----
    # Try swapping elements in a row to reduce its maximum while keeping validity.
    if time.time() < deadline - 10:
        for r in range(n_rows):
            if time.time() > deadline - 10:
                break
            row = rows[r]
            cur_max = row[-1]
            used_without = set(used)
            used_without.difference_update(row_diffs(row))

            # Try to find a better row by local search
            improved_row = None
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                # Start from current row and try to reduce the last element
                cand_row = row[:]
                for pos in range(len(cand_row) - 2, -1, -1):
                    if time.time() > deadline - 10:
                        break
                    # Try to reduce cand_row[pos]
                    old_val = cand_row[pos]
                    for new_val in range(old_val - 1, max(cand_row[pos-1] if pos > 0 else -1, -1), -1):
                        if time.time() > deadline - 10:
                            break
                        test_row = cand_row[:]
                        test_row[pos] = new_val
                        # Check if this row is valid
                        test_diffs = row_diffs(test_row)
                        if not (test_diffs & used_without):
                            # Also check that the row is strictly increasing
                            if all(test_row[i] < test_row[i+1] for i in range(len(test_row)-1)):
                                # Check that this new row doesn't conflict with other rows
                                # (already checked via used_without)
                                # Update cand_row and continue
                                cand_row = test_row
                                break
                # If we improved the row, update and break
                if cand_row[-1] < cur_max:
                    improved_row = cand_row
                    break

            if improved_row is not None and improved_row[-1] < cur_max:
                used.difference_update(row_diffs(rows[r]))
                rows[r] = improved_row
                used |= row_diffs(improved_row)

    # ---- Phase 5: randomized restarts with perturbation ----
    # Try a few randomized constructions to find a better local optimum
    best_rows = rows[:]
    best_max = max(row[-1] for row in rows)

    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            # Randomized greedy construction
            used_r = set()
            rows_r = []
            for i in range(n_rows):
                # Try a few random perturbations
                best_row = None
                best_last = float('inf')
                for _ in range(3):
                    row = greedy_row(set(used_r), k, randomize=True)
                    if row is not None and row[-1] < best_last:
                        best_row = row
                        best_last = row[-1]
                if best_row is None:
                    best_row = greedy_row(set(used_r), k)
                rows_r.append(best_row)
                used_r |= row_diffs(best_row)

            cur_max = max(row[-1] for row in rows_r)
            if cur_max < best_max:
                best_rows = rows_r
                best_max = cur_max
                used = used_r
                rows = rows_r

                # Run improvement phases on the new best
                if time.time() < deadline - 10:
                    order2 = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
                    improved2 = True
                    while improved2 and time.time() < deadline - 5:
                        improved2 = False
                        for r in order2:
                            if time.time() > deadline - 5:
                                break
                            cur_last = rows[r][-1]
                            best = None
                            for cap in range(cur_last - 1, max(cur_last - 30, 0), -1):
                                if time.time() > deadline - 5:
                                    break
                                used_without = set(used)
                                used_without.difference_update(row_diffs(rows[r]))
                                row_deadline = min(deadline - 5, time.time() + 2)
                                cand = dfs_row([0], used_without, cap, row_deadline)
                                if cand is not None:
                                    best = cand
                                    break
                            if best is not None and best[-1] < cur_last:
                                used.difference_update(row_diffs(rows[r]))
                                rows[r] = best
                                used |= row_diffs(best)
                                improved2 = True

    return rows
