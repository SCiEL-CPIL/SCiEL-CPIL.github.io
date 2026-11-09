"""Difference triangle set (7,5) – interleaved greedy + simulated annealing + backtracking."""
import time
import random
import sys
import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

_START = time.time()
_DEADLINE = _START + 142.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _row_diffs(row):
    """All positive differences a_j - a_i for i<j in one row."""
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
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    """Build rows interleaved: at each step, extend the row with the smallest
    current maximum. This tends to keep all rows balanced and compact."""
    rows = [[0] for _ in range(n_rows)]
    used = set()

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                found = True
                break
            x += 1
        if not found:
            return None, None

    for r in rows:
        if len(r) != k + 1:
            return None, None
    return rows, used


def _greedy_row(used, k, max_mark, rnd=None):
    """Greedy place one row of length k+1 starting at 0; smallest marks that keep
    all differences unused. Returns (row, used_delta, ok)."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        new = []
        bad = False
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _greedy_fill(n_rows, k, max_mark, rnd=None):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, local, ok = _greedy_row(used, k, max_mark, rnd)
        if not ok:
            return None, None
        rows.append(row)
        used |= local
    return rows, used


def _simulated_annealing(rows, n_rows, k, time_budget, rnd):
    """Simulated annealing to reduce scope. Moves: replace one row with a new
    random/greedy row that keeps the packing valid."""
    if rows is None:
        return None, None

    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    # Precompute used set
    def compute_used(rows):
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        return used

    used = compute_used(current)

    deadline = time.time() + time_budget
    T = 50.0
    cooling = 0.995
    step = 0

    while time.time() < deadline and T > 0.1:
        step += 1
        if step % 1000 == 0 and _dead():
            break

        # Pick a random row to replace
        ri = rnd.randint(0, n_rows - 1)
        old_row = current[ri]

        # Generate a new row: try greedy with slight randomization
        # Remove old row's diffs from used
        old_diffs = _row_diffs(old_row)
        used_without = used - old_diffs

        # Try to build a new row with max_mark = current_scope - 1 (or best_scope - 1)
        max_mark = min(current_scope, best_scope) - 1
        if max_mark < k:
            max_mark = best_scope

        # Try multiple randomizations
        new_row = None
        for attempt in range(20):
            # Slightly randomized greedy
            row = [0]
            local = set()
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                # Sometimes skip a value to randomize
                if rnd.random() < 0.1 and x < max_mark:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_without or dv in local:
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
            # Try a completely random row
            row = [0]
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > max_mark:
                    success = False
                    break
                new = []
                bad = False
                for a in row:
                    dv = x - a
                    if dv in used_without or dv in local:
                        bad = True
                        break
                    new.append(dv)
                if not bad:
                    row.append(x)
                    local.update(new)
            if success and len(row) == k + 1:
                new_row = row

        if new_row is None:
            T *= cooling
            continue

        # Check validity of new row
        new_diffs = _row_diffs(new_row)
        if new_diffs & used_without:
            T *= cooling
            continue

        # Compute new scope
        new_scope = max(max(r) for i, r in enumerate(current) if i != ri)
        new_scope = max(new_scope, max(new_row))

        # Acceptance criterion
        delta = new_scope - current_scope
        if delta <= 0 or rnd.random() < math.exp(-delta / T):
            # Accept
            used = used_without | new_diffs
            current[ri] = new_row
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [r[:] for r in current]
        else:
            # Reject
            pass

        T *= cooling

    return best, best_scope


def _backtrack_search(n_rows, k, target, rnd, time_budget=3.0):
    """Backtracking search for a valid packing into [0..target]."""
    rows = [None] * n_rows
    used = set()

    def fill_row(row_id, marks, used_set):
        """Recursively fill one row."""
        if time.time() > time_budget:
            return False
        if len(marks) == k + 1:
            rows[row_id] = [0] + list(marks)
            return True
        start = (marks[-1] + 1) if marks else 1
        for x in range(start, target + 1):
            prev = [0] + list(marks)
            newl = set()
            bad = False
            for a in prev:
                dv = x - a
                if dv in used_set or dv in newl:
                    bad = True
                    break
                newl.add(dv)
            if bad:
                continue
            marks.append(x)
            used_set |= newl
            if fill_row(row_id, marks, used_set):
                return True
            marks.pop()
            used_set -= newl
        return False

    for i in range(n_rows):
        if not fill_row(i, [], used):
            return None, None
    return rows, used


def construct(n_rows, k):
    """Main entry point: find the smallest possible scope for n_rows rows of k+1
    increasing integers starting at 0, with all positive differences distinct."""
    rnd = random.Random(42)
    
    # Phase 1: Interleaved greedy to get a good initial solution
    # Start with a reasonable max_mark estimate
    # The counting bound suggests scope >= 105, known best is 112
    # Try interleaved greedy with increasing max_mark until we find a solution
    
    best_rows = None
    best_scope = float('inf')
    
    # Try interleaved greedy with various max_mark values
    for max_mark in range(k + n_rows, 130):
        if _dead():
            break
        rows, used = _interleaved_greedy(n_rows, k, max_mark, rnd)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            # If we found a solution, we can stop early if it's good enough
            if scope <= 112:
                break
    
    # If interleaved greedy didn't find anything, try standard greedy
    if best_rows is None:
        for max_mark in range(k + n_rows, 130):
            if _dead():
                break
            rows, used = _greedy_fill(n_rows, k, max_mark, rnd)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                if scope <= 112:
                    break
    
    if best_rows is None:
        # Fallback: return a trivially valid (but suboptimal) solution
        # This shouldn't happen for reasonable n_rows, k
        used = set()
        rows = []
        for _ in range(n_rows):
            row, local, ok = _greedy_row(used, k, 200, rnd)
            if ok:
                rows.append(row)
                used |= local
            else:
                # Should not happen
                row = [0] + list(range(1, k + 1))
                rows.append(row)
        return rows
    
    # Phase 2: Simulated annealing to improve the solution
    # Use remaining time budget
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 5:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, n_rows, k, time_remaining - 10, rnd
        )
        if improved_rows is not None and improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    # Phase 3: Try backtracking with target = best_scope - 1 to see if we can do better
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 5 and best_scope > k + 1:
        target = best_scope - 1
        bt_rows, _ = _backtrack_search(n_rows, k, target, rnd, time_budget=min(time_remaining - 5, 30))
        if bt_rows is not None:
            bt_scope = _scope(bt_rows)
            if bt_scope < best_scope:
                best_rows = bt_rows
                best_scope = bt_scope
    
    return best_rows
