"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
from collections import defaultdict

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy_construct(n_rows, k, used, rng):
    """Build one row greedily, optionally with small randomization."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row


def _backtrack_construct(n_rows, k, max_scope, rng):
    """
    Try to construct n_rows rows with scope <= max_scope using backtracking.
    Returns list of rows or None.
    """
    used = set()
    rows = []

    def build_row(depth, current_row, current_max):
        if depth == k + 1:
            rows.append(current_row[:])
            return True
        if depth == 0:
            # Start with 0
            if not build_row(1, [0], 0):
                return False
            return True
        # Try next values starting from last + 1
        start = current_row[-1] + 1
        for x in range(start, max_scope + 1):
            new = {x - a for a in current_row}
            if new & used:
                continue
            # Check scope
            if x > max_scope:
                break
            # Try this
            for d in new:
                used.add(d)
            current_row.append(x)
            if build_row(depth + 1, current_row, x):
                return True
            current_row.pop()
            for d in new:
                used.discard(d)
        return False

    for _ in range(n_rows):
        if not build_row(0, [], 0):
            return None
    return rows


def _local_search(rows, n_rows, k, max_iters=50000):
    """
    Local search: try to reduce the largest entry by moving marks.
    We focus on the row with the maximum entry and try to shrink it.
    """
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0

    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        # Find the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        current_max = row[-1]

        if current_max <= best_scope:
            break

        # Try to reduce the last element
        improved = False
        # Try moving the last mark down
        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            # Check internal differences
            internal_ok = True
            internal_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in internal_diffs:
                        internal_ok = False
                        break
                    internal_diffs.add(d)
            if not internal_ok:
                continue
            # Check against other rows
            other_diffs = set()
            for r2 in rows:
                if r2 is row:
                    continue
                for i in range(len(r2)):
                    for j in range(i + 1, len(r2)):
                        other_diffs.add(r2[j] - r2[i])
            if internal_diffs & other_diffs:
                continue
            # Valid! Update
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break

        if not improved:
            # Try reducing other marks in the max row
            found_move = False
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                # Try reducing this mark
                lower_bound = row[pos - 1] + 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound:
                        break
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    # Check strictly increasing
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal_diffs = set()
                    internal_ok = True
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in internal_diffs:
                                internal_ok = False
                                break
                            internal_diffs.add(d)
                    if not internal_ok:
                        continue
                    other_diffs = set()
                    for r2 in rows:
                        if r2 is row:
                            continue
                        for i in range(len(r2)):
                            for j in range(i + 1, len(r2)):
                                other_diffs.add(r2[j] - r2[i])
                    if internal_diffs & other_diffs:
                        continue
                    # Valid
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break
            if not found_move:
                # Try random perturbation then reduction
                if rng.random() < 0.3:
                    # Randomly shuffle order of trying reductions
                    pass
                # If stuck, try a small random restart of the max row
                if rng.random() < 0.1 and _time_left() > 20.0:
                    # Try rebuilding the max row with a slightly different strategy
                    used_other = set()
                    for r2 in rows:
                        if r2 is row:
                            continue
                        for i in range(len(r2)):
                            for j in range(i + 1, len(r2)):
                                used_other.add(r2[j] - r2[i])
                    # Try greedy with randomization
                    new_row = [0]
                    x = 0
                    while len(new_row) < k + 1:
                        x += 1
                        if x > best_scope:
                            break
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & used_other) and not (new_diffs & set()):
                            # Check internal
                            internal = set()
                            ok = True
                            for i in range(len(new_row)):
                                for j in range(i + 1, len(new_row)):
                                    d = new_row[j] - new_row[i]
                                    if d in internal:
                                        ok = False
                                        break
                                    internal.add(d)
                            if ok and not (new_diffs & internal):
                                new_row.append(x)
                    if len(new_row) == k + 1 and new_row[-1] <= best_scope:
                        rows[max_idx] = new_row
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope

    return rows, best_scope


def _multi_pass_reduction(rows, n_rows, k, rng):
    """
    Multi-pass reduction: repeatedly try to reduce the global max scope.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    passes = 0
    max_passes = 50

    while passes < max_passes and _time_left() > 10.0:
        passes += 1
        rows = [r[:] for r in best_rows]
        # Try to reduce each row's max
        improved_any = False
        for r_idx in range(n_rows):
            row = rows[r_idx]
            current_max = row[-1]
            if current_max < best_scope:
                continue
            # Try to reduce
            for new_last in range(current_max - 1, row[-2], -1):
                if new_last <= row[-2]:
                    break
                new_row = row[:-1] + [new_last]
                internal_diffs = set()
                internal_ok = True
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in internal_diffs:
                            internal_ok = False
                            break
                        internal_diffs.add(d)
                if not internal_ok:
                    continue
                other_diffs = set()
                for r2 in rows:
                    if r2 is row:
                        continue
                    for i in range(len(r2)):
                        for j in range(i + 1, len(r2)):
                            other_diffs.add(r2[j] - r2[i])
                if internal_diffs & other_diffs:
                    continue
                rows[r_idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                improved_any = True
                break
        if not improved_any:
            break

    return best_rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)

    # Phase 1: Greedy construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_construct(n_rows, k, used, rng)
        rows.append(row)

    if not _validate(rows):
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)

    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]

    # Phase 2: Local search
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search(rows, n_rows, k)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 3: Multi-pass reduction
    if _time_left() > 10.0:
        rows_mp, scope_mp = _multi_pass_reduction(best_rows, n_rows, k, rng)
        if scope_mp < best_scope:
            best_scope = scope_mp
            best_rows = rows_mp

    # Phase 4: Additional local search passes
    if _time_left() > 8.0:
        rows_ls2, scope_ls2 = _local_search(best_rows, n_rows, k, max_iters=20000)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2

    # Final validation
    if not _validate(best_rows):
        # Return the greedy result
        return rows

    return best_rows
