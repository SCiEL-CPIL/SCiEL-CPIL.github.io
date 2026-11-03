"""Difference triangle set (7,5): minimize largest entry with distinct cross-row differences.

Design (greedy family, refined):
  - Build rows greedily, choosing the next mark to keep the row tight (minimize span growth),
    breaking ties toward the smallest mark.
  - After an initial full build, run a local "tighten" pass: rebuild each row (in turn) against
    the used-difference set that excludes that row's own differences, minimizing that row's
    largest entry.  Accept a rebuild only if the global scope (max entry over all rows) does not
    increase.
  - Repeat tighten passes under an explicit wall-clock deadline, keeping the best complete solution.
  - Return the best solution with a safety margin before the 150 s kill.
"""

import time
import random

# EVOLVE-BLOCK-START

_DEADLINE_FRAC = 0.90          # stop searching at 90% of the 150 s budget
_HARD_LIMIT = 150.0            # wall-clock seconds


def _row_diffs(row):
    """All positive differences a[j]-a[i] for i<j inside a single row."""
    diffs = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ai)
    return diffs


def _max_entry(rows):
    m = 0
    for r in rows:
        if r:
            m = max(m, r[-1])
    return m


def _build_row(used, k, rng, lookahead=2):
    """Greedily build one row of k+1 increasing ints starting at 0.

    At each step, consider candidate marks x (in increasing order) whose differences to the
    current partial row are all unused.  Among the first few feasible candidates, pick the one
    that minimizes a tightness surrogate (the running span, i.e. the candidate mark itself,
    since the row starts at 0 the span equals the last mark).  This keeps rows compact, which
    is what ultimately bounds the global scope.
    """
    row = [0]
    while len(row) < k + 1:
        best_x = None
        best_key = None
        x = row[-1] + 1
        considered = 0
        # Scan candidates in increasing order; stop early once we've seen enough feasible ones
        # and the span is clearly no longer improving (span is monotone in x, so the first
        # feasible x is the tightest; we only need the first feasible x).
        while True:
            new = {x - a for a in row}
            if not (new & used):
                # First feasible x is the smallest -> tightest possible next mark.
                best_x = x
                break
            x += 1
            considered += 1
            if considered > 200000:  # safety; should not happen
                break
        if best_x is None:
            # Fallback: keep bumping (should not occur for reasonable inputs)
            best_x = x + 1
        row.append(best_x)
        used.update({best_x - a for a in row[:-1]})
    return row


def _tighten_row(used_without_row, k, current_row):
    """Rebuild a single row against the used set that excludes its own differences,
    minimizing the row's largest entry.  Returns the new row (or None if not strictly better)."""
    # Rebuild greedily (first-feasible = tightest).
    new_row = _build_row(set(used_without_row), k, None)
    if new_row[-1] < current_row[-1]:
        return new_row
    return None


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _HARD_LIMIT * _DEADLINE_FRAC
    rng = random.Random(12345)

    # ---- Phase 1: initial greedy build ------------------------------------
    used = set()
    rows = []
    for _ in range(n_rows):
        r = _build_row(used, k, rng)
        used.update(_row_diffs(r))
        rows.append(r)

    best_rows = [r[:] for r in rows]
    best_scope = _max_entry(best_rows)

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    # ---- Phase 2: iterative tighten passes --------------------------------
    # Repeatedly rebuild each row (in a rotating order) against the used set that
    # excludes that row's own differences; accept if global scope does not increase.
    pass_count = 0
    while _elapsed_ok():
        pass_count += 1
        order = list(range(n_rows))
        if pass_count % 2 == 0:
            order.reverse()
        improved = False
        for idx in order:
            if not _elapsed_ok():
                break
            # used set excluding this row's differences
            others = set()
            for j in range(n_rows):
                if j != idx:
                    others.update(_row_diffs(rows[j]))
            new_row = _tighten_row(others, k, rows[idx])
            if new_row is not None:
                # Rebuild global used set and check scope
                rows[idx] = new_row
                new_scope = _max_entry(rows)
                if new_scope <= best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    improved = True
                else:
                    # revert (should not happen since new_row is tighter, but be safe)
                    pass
        if not improved and pass_count >= 3:
            # A couple of extra passes to be sure we've converged, then stop.
            if pass_count >= 4:
                break

    return best_rows

# EVOLVE-BLOCK-END
