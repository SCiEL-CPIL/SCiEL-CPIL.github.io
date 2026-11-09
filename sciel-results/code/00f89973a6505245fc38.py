"""Difference triangle set (7,5) – constraint-satisfaction with backtracking + bitmask."""
import time
import random
import sys
import math
from collections import Counter

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

_START = time.time()
_DEADLINE = _START + 145.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    d = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _greedy_construct(n_rows, k):
    """Build rows greedily, each row uses smallest possible marks."""
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


def _interleaved_greedy(n_rows, k):
    """Build all rows simultaneously, adding one mark at a time to the row with smallest max."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        while True:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                break
            x += 1
    return rows


def _backtrack_with_mrv(n_rows, k, max_mark, rnd, time_limit=5.0):
    """
    Backtracking search with MRV heuristic.
    Tries to fill all rows within [0, max_mark].
    Uses forward checking to prune early.
    """
    rows = [None] * n_rows
    used_mask = 0  # bitmask of used differences
    max_diff = max_mark  # differences can be at most max_mark

    def mask_of_diffs(diffs):
        m = 0
        for d in diffs:
            if d <= max_diff:
                m |= (1 << d)
        return m

    def row_diffs_mask(row):
        return mask_of_diffs(_diffs_of_row(row))

    def check_row_valid(row, used_m, local_m):
        """Check if row's diffs don't conflict with used or local."""
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d <= max_diff:
                    if (used_m >> d) & 1 or (local_m >> d) & 1:
                        return False
        return True

    def backtrack(row_idx, marks, local_mask):
        if time.time() > time_limit:
            return False
        if len(marks) == k + 1:
            rows[row_idx] = [0] + list(marks)
            return True

        prev = [0] + list(marks)
        start = (marks[-1] + 1) if marks else 1

        # MRV: try values that leave the most options for future
        # For simplicity, try from start upward but with some randomization
        candidates = []
        for x in range(start, max_mark + 1):
            # Check if x can be added
            new_local = local_mask
            ok = True
            for a in prev:
                d = x - a
                if d <= max_diff:
                    if (used_mask >> d) & 1 or (new_local >> d) & 1:
                        ok = False
                        break
                    new_local |= (1 << d)
            if ok:
                candidates.append((x, new_local))

        if not candidates:
            return False

        # Shuffle for randomization
        if rnd.random() < 0.3:
            rnd.shuffle(candidates)

        for x, new_local in candidates:
            marks.append(x)
            if backtrack(row_idx, marks, new_local):
                return True
            marks.pop()

        return False

    for i in range(n_rows):
        if not backtrack(i, [], 0):
            return None
        # Update used_mask
        row_diffs = _diffs_of_row(rows[i])
        for d in row_diffs:
            if d <= max_diff:
                used_mask |= (1 << d)

    return rows


def _iterative_deepening_backtrack(n_rows, k, initial_max, rnd, total_time=100.0):
    """
    Start from a greedy solution's scope, then try to find solutions with smaller scope.
    Use backtracking with MRV to find valid packings.
    """
    # First get a greedy solution
    greedy = _greedy_construct(n_rows, k)
    best_rows = greedy
    best_scope = _scope(greedy)

    deadline = time.time() + total_time

    # Try decreasing scope
    current_max = best_scope - 1
    while current_max >= k and time.time() < deadline - 5:
        # Try backtracking with this max
        time_per_try = min(3.0, max(0.5, (deadline - time.time()) * 0.1))
        result = _backtrack_with_mrv(n_rows, k, current_max, rnd, time_limit=time_per_try)
        if result is not None:
            # Verify
            if _valid(result, n_rows, k):
                sc = _scope(result)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = result
                    current_max = sc - 1
                    continue
        current_max -= 1

    return best_rows, best_scope


def _local_search_improved(initial_rows, n_rows, k, time_limit=120.0):
    """
    Improved local search:
    - Try to replace rows with better ones
    - Use conflict counting
    - Accept moves that reduce conflicts or scope
    """
    best_rows = [r[:] for r in initial_rows]
    best_scope = _scope(best_rows)

    current_rows = [r[:] for r in initial_rows]
    current_scope = _scope(current_rows)

    def count_conflicts(rows):
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        c = Counter(all_diffs)
        return sum(v - 1 for v in c.values() if v > 1)

    current_conflicts = count_conflicts(current_rows)
    start_time = time.time()
    no_improve = 0
    max_no_improve = 50000

    while time.time() < time_limit and no_improve < max_no_improve:
        # Pick a random row
        r_idx = rnd.randint(0, n_rows - 1)
        old_row = list(current_rows[r_idx])

        # Try to generate a new row
        # Remove old row's diffs
        old_diffs = _diffs_of_row(old_row)
        other_diffs = set()
        for i, row in enumerate(current_rows):
            if i != r_idx:
                other_diffs |= _diffs_of_row(row)

        # Try greedy construction with randomization
        new_row = None
        for attempt in range(10):
            row = [0]
            local = set()
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > current_scope:
                    success = False
                    break
                # Random skip
                if rnd.random() < 0.15 and x < current_scope:
                    x += 1
                    if x > current_scope:
                        success = False
                        break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in other_diffs or dv in local:
                        bad = True
                        break
                    new.append(dv)
                if not bad:
                    row.append(x)
                    local.update(new)
            if success and len(row) == k + 1:
                new_row = row
                break

        if new_row is None:
            no_improve += 1
            continue

        new_diffs = _diffs_of_row(new_row)
        conflict_count = len(new_diffs & other_diffs)

        temp_rows = list(current_rows)
        temp_rows[r_idx] = new_row
        new_scope = _scope(temp_rows)

        if conflict_count < current_conflicts:
            current_rows[r_idx] = new_row
            current_conflicts = conflict_count
            current_scope = new_scope
            no_improve = 0
            if current_conflicts == 0 and current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
        elif conflict_count == current_conflicts:
            if new_scope < current_scope:
                current_rows[r_idx] = new_row
                current_scope = new_scope
                no_improve = 0
                if current_conflicts == 0 and current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                no_improve += 1
        else:
            no_improve += 1

    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    rnd = random.Random(42)

    deadline = time.time() + 140.0

    # Phase 1: Get initial solutions
    greedy = _greedy_construct(n_rows, k)
    interleaved = _interleaved_greedy(n_rows, k)

    best_rows = greedy
    best_scope = _scope(greedy)

    if _scope(interleaved) < best_scope:
        best_scope = _scope(interleaved)
        best_rows = interleaved

    # Phase 2: Try iterative deepening backtracking to find smaller scope
    if time.time() < deadline - 30:
        bt_rows, bt_scope = _iterative_deepening_backtrack(
            n_rows, k, best_scope, rnd, total_time=min(30.0, deadline - time.time() - 10)
        )
        if bt_scope < best_scope:
            best_scope = bt_scope
            best_rows = bt_rows

    # Phase 3: Local search to further improve
    if time.time() < deadline - 10:
        time_left = deadline - time.time() - 5
        ls_rows, ls_scope = _local_search_improved(
            best_rows, n_rows, k, time_limit=min(60.0, time_left)
        )
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows

    # Phase 4: Try more backtracking if time remains
    if time.time() < deadline - 5 and best_scope > k:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            bt_rows, bt_scope = _iterative_deepening_backtrack(
                n_rows, k, best_scope, rnd, total_time=min(20.0, remaining)
            )
            if bt_scope < best_scope:
                best_scope = bt_scope
                best_rows = bt_rows

    # Final validation
    if not _valid(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = _greedy_construct(n_rows, k)

    return best_rows
