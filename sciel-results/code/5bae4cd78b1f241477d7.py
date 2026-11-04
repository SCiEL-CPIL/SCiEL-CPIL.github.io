"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = None
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    def full_row_diffs(row_vals):
        """Row includes 0 at start. So diffs include all row_vals and pairwise diffs."""
        diffs = set()
        for v in row_vals:
            diffs.add(v)
        n = len(row_vals)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row_vals[j] - row_vals[i])
        return diffs

    def validate(rows, n_rows, k):
        """Validate that all differences are distinct across all rows."""
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k:
                return False
            prev = 0
            for v in row:
                if v <= prev:
                    return False
                prev = v
            diffs = full_row_diffs(row)
            if diffs & used:
                return False
            used |= diffs
        return True

    def backtrack_construct(n_rows, k, max_scope, rng, time_budget=2.0):
        """
        Backtracking construction with forward checking.
        Tries to build n_rows rows where all entries <= max_scope.
        Returns list of rows (each with k non-zero values) or None.
        """
        start_time = time.time()
        used = set()
        rows = []
        row_idx = 0
        elem_idx = 0

        # Use iterative deepening on scope if needed, but here we fix max_scope
        # We'll use a recursive approach with explicit stack for speed

        def try_build():
            nonlocal used, rows, row_idx, elem_idx
            if time.time() - start_time > time_budget:
                return False

            if row_idx == n_rows:
                return True

            row = rows[row_idx]
            if elem_idx == k:
                # Row complete
                rows.append(row)
                row_idx += 1
                elem_idx = 0
                return try_build()

            # Determine lower bound for next value
            if elem_idx == 0:
                lo = 1
            else:
                lo = row[-1] + 1

            hi = max_scope

            # Forward checking: for each candidate, check if it's feasible
            found = False
            for cand in range(lo, hi + 1):
                # Check new diffs
                new_diffs = {cand}
                for v in row:
                    new_diffs.add(cand - v)

                if new_diffs & used:
                    continue

                # Tentatively add
                old_used = used
                used = old_used | new_diffs
                row.append(cand)
                elem_idx += 1

                if try_build():
                    found = True
                    break

                # Undo
                row.pop()
                elem_idx -= 1
                used = old_used

            return found

        # Start building
        rows = []
        row_idx = 0
        elem_idx = 0
        current_row = []
        rows.append(current_row)

        if try_build():
            return rows
        return None

    def greedy_with_pruning(n_rows, k, rng, bias=0.7, window=40, max_cands=20, max_scope=None):
        """
        Greedy construction with forward-checking pruning.
        For each element, only consider candidates that don't immediately block
        future elements (heuristic check).
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = []
            x = 0
            while len(row) < k:
                valid_cands = []
                search_limit = x + window
                if max_scope:
                    search_limit = min(search_limit, max_scope)
                for candidate in range(x + 1, search_limit + 1):
                    new_diffs = {candidate} | {candidate - v for v in row}
                    if not (new_diffs & used):
                        valid_cands.append(candidate)
                        if len(valid_cands) >= max_cands:
                            break
                if not valid_cands:
                    candidate = x + 1
                    while True:
                        if max_scope and candidate > max_scope:
                            break
                        new_diffs = {candidate} | {candidate - v for v in row}
                        if not (new_diffs & used):
                            valid_cands.append(candidate)
                            break
                        candidate += 1
                    if not valid_cands:
                        return None  # Failed

                if rng.random() < bias:
                    x = valid_cands[0]
                else:
                    x = rng.choice(valid_cands)

                row.append(x)
                used |= {x} | {x - v for v in row[:-1]}
            rows.append(row)
        return rows

    def multi_local_search(rows, k, max_iters=200):
        """
        More aggressive local search: try to reduce any element of any row,
        not just the last element.
        """
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            used |= full_row_diffs(row)

        improved_any = False
        for _ in range(max_iters):
            if _time_left() < 3.0:
                break

            # Try to reduce the largest entry
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            max_val = row[-1]

            # Try reducing last element
            found = False
            for new_val in range(row[-2] + 1, max_val):
                old_diffs = full_row_diffs(row)
                new_row = row[:-1] + [new_val]
                new_diffs = full_row_diffs(new_row)
                added_diffs = new_diffs - old_diffs
                if added_diffs & (used - old_diffs):
                    continue
                used -= old_diffs
                used |= new_diffs
                row[-1] = new_val
                found = True
                improved_any = True
                break

            if not found:
                # Try reducing other elements in the max row
                for pos in range(k - 2, -1, -1):
                    old_val = row[pos]
                    lo = row[pos - 1] + 1 if pos > 0 else 1
                    for new_val in range(lo, old_val):
                        old_diffs = full_row_diffs(row)
                        new_row = row[:pos] + [new_val] + row[pos + 1:]
                        # Check strictly increasing
                        if pos > 0 and new_val <= row[pos - 1]:
                            continue
                        if pos < k - 1 and new_val >= row[pos + 1]:
                            continue
                        new_diffs = full_row_diffs(new_row)
                        added_diffs = new_diffs - old_diffs
                        if added_diffs & (used - old_diffs):
                            continue
                        used -= old_diffs
                        used |= new_diffs
                        row[pos] = new_val
                        found = True
                        improved_any = True
                        break
                    if found:
                        break

                if not found:
                    # Try other rows
                    found_any_in_pass = False
                    for idx in range(len(rows)):
                        if idx == max_idx:
                            continue
                        row = rows[idx]
                        for pos in range(k - 1, -1, -1):
                            old_val = row[pos]
                            lo = row[pos - 1] + 1 if pos > 0 else 1
                            for new_val in range(lo, old_val):
                                old_diffs = full_row_diffs(row)
                                new_row = row[:pos] + [new_val] + row[pos + 1:]
                                if pos > 0 and new_val <= row[pos - 1]:
                                    continue
                                if pos < k - 1 and new_val >= row[pos + 1]:
                                    continue
                                new_diffs = full_row_diffs(new_row)
                                added_diffs = new_diffs - old_diffs
                                if added_diffs & (used - old_diffs):
                                    continue
                                used -= old_diffs
                                used |= new_diffs
                                row[pos] = new_val
                                found_any_in_pass = True
                                improved_any = True
                                break
                            if found_any_in_pass:
                                break
                        if found_any_in_pass:
                            break
                    if not found_any_in_pass:
                        break

        if improved_any:
            return rows
        return None

    # --- Main search strategy ---
    rng = random.Random(42)

    # Lower bound
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))

    # Start with a greedy solution
    best_rows = greedy_with_pruning(n_rows, k, rng, bias=1.0, window=50, max_cands=25)
    if best_rows is None:
        # Fallback: very greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = []
            x = 0
            while len(row) < k:
                cand = x + 1
                while True:
                    new_diffs = {cand} | {cand - v for v in row}
                    if not (new_diffs & used):
                        break
                    cand += 1
                row.append(cand)
                used |= {cand} | {cand - v for v in row[:-1]}
            best_rows.append(row)

    best_max = max(max(r) for r in best_rows)

    # Strategy 1: Multiple greedy restarts with varying parameters
    for attempt in range(60):
        if _time_left() < 5.0:
            break
        bias = 0.4 + 0.5 * rng.random()
        window = 25 + int(40 * rng.random())
        max_cands = 10 + int(25 * rng.random())
        rows = greedy_with_pruning(n_rows, k, rng, bias=bias, window=window, max_cands=max_cands)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            if _time_left() > 5.0:
                improved = multi_local_search(best_rows, k, max_iters=100)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = improved

    # Strategy 2: Backtracking with tight scope for small cases
    if n_rows * k <= 30 and _time_left() > 10.0:
        # Try backtracking with scope = best_max - 1, then best_max - 2, etc.
        for scope_offset in range(1, min(15, best_max - lower_bound + 1)):
            if _time_left() < 5.0:
                break
            target_scope = best_max - scope_offset
            if target_scope < lower_bound:
                break
            bt_rows = backtrack_construct(n_rows, k, target_scope, rng, time_budget=3.0)
            if bt_rows is not None:
                best_max = target_scope
                best_rows = bt_rows
                break

    # Strategy 3: Final aggressive local search
    if _time_left() > 3.0:
        improved = multi_local_search(best_rows, k, max_iters=300)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # Convert to full rows
    full_rows = [[0] + row for row in best_rows]

    # Validate
    if not validate(full_rows, n_rows, k):
        # Fallback
        used = set()
        fallback_rows = []
        for _ in range(n_rows):
            row = []
            x = 0
            while len(row) < k:
                cand = x + 1
                while True:
                    new_diffs = {cand} | {cand - v for v in row}
                    if not (new_diffs & used):
                        break
                    cand += 1
                row.append(cand)
                used |= {cand} | {cand - v for v in row[:-1]}
            fallback_rows.append(row)
        full_rows = [[0] + row for row in fallback_rows]

    return full_rows
