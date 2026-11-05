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


def _validate(rows, n_rows, k):
    """Check that rows form a valid difference triangle set."""
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


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(row, used, k, rng=None):
    """
    Complete a partial row (starting with [0]) to length k+1 using greedy.
    Returns the completed row or None if stuck.
    """
    row = list(row)
    x = row[-1]
    while len(row) < k + 1:
        # Find the smallest valid next value
        found = False
        for candidate in range(x + 1, x + 200):
            new_diffs = {candidate - a for a in row}
            if not new_diffs & used:
                row.append(candidate)
                used |= new_diffs
                x = candidate
                found = True
                break
        if not found:
            return None
    return row


def _greedy_row_randomized(row, used, k, rng):
    """
    Complete a partial row with randomized choices to diversify.
    Returns the completed row or None if stuck.
    """
    row = list(row)
    x = row[-1]
    while len(row) < k + 1:
        # Collect valid candidates in a window
        valid = []
        for candidate in range(x + 1, x + 60):
            new_diffs = {candidate - a for a in row}
            if not new_diffs & used:
                valid.append(candidate)
                if len(valid) >= 15:
                    break
        if not valid:
            # Extend search
            candidate = x + 1
            while True:
                new_diffs = {candidate - a for a in row}
                if not new_diffs & used:
                    valid.append(candidate)
                    break
                candidate += 1

        # Choose with bias toward smaller
        if rng.random() < 0.6:
            x = valid[0]
        else:
            x = rng.choice(valid)
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row


def _construct(n_rows, k, rng, mode='greedy'):
    """
    Construct n_rows rows. mode='greedy' or 'randomized'.
    Returns (rows, used_set) or (None, None) on failure.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        if mode == 'greedy':
            result = _greedy_row(row, used, k)
        else:
            result = _greedy_row_randomized(row, used, k, rng)
        if result is None:
            return None, None
        rows.append(result)
    return rows, used


def _rebuild_row(rows, used, row_idx, k, rng, mode='greedy'):
    """
    Rebuild row at row_idx by removing it, then reconstructing it.
    Returns (new_rows, new_used) or (None, None) on failure.
    """
    # Remove old row's differences
    old_row = rows[row_idx]
    old_diffs = set()
    for i in range(len(old_row)):
        for j in range(i + 1, len(old_row)):
            old_diffs.add(old_row[j] - old_row[i])
    used -= old_diffs

    # Rebuild
    row = [0]
    if mode == 'greedy':
        result = _greedy_row(row, used, k)
    else:
        result = _greedy_row_randomized(row, used, k, rng)

    if result is None:
        # Restore old state
        used |= old_diffs
        return None, None

    new_rows = [list(r) for r in rows]
    new_rows[row_idx] = result
    used |= {result[j] - result[i] for i in range(len(result)) for j in range(i + 1, len(result))}
    return new_rows, used


def _iterative_tightening(rows, k, rng, max_rounds=200):
    """
    Iteratively try to reduce the scope by rebuilding rows that have the largest entries.
    Returns improved rows or the original if no improvement.
    """
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])

    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)

    for rnd in range(max_rounds):
        if _time_left() < 3.0:
            break

        # Find the row with the largest last element
        max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        # Try to rebuild this row with a tighter constraint
        # First, try greedy rebuild
        new_rows, new_used = _rebuild_row(rows, used, max_idx, k, rng, mode='greedy')
        if new_rows is not None:
            new_scope = _scope(new_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = new_rows
                rows = new_rows
                used = new_used
            elif new_scope == best_scope and rng.random() < 0.3:
                # Keep as alternative for diversity
                rows = new_rows
                used = new_used
        else:
            # Try randomized rebuild
            new_rows, new_used = _rebuild_row(rows, used, max_idx, k, rng, mode='randomized')
            if new_rows is not None:
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = new_rows
                    rows = new_rows
                    used = new_used

        # Also try rebuilding other random rows to escape local optima
        if rnd % 5 == 0:
            other_idx = rng.randint(0, len(rows) - 1)
            if other_idx != max_idx:
                new_rows, new_used = _rebuild_row(rows, used, other_idx, k, rng, mode='randomized')
                if new_rows is not None:
                    new_scope = _scope(new_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = new_rows
                        rows = new_rows
                        used = new_used

    return best_rows


def _local_search(rows, k, rng, max_iters=300):
    """
    Local search: try to reduce individual entries by finding smaller valid replacements.
    """
    rows = [list(r) for r in rows]
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])

    improved = True
    iterations = 0
    while improved and iterations < max_iters:
        if _time_left() < 2.0:
            break
        improved = False
        iterations += 1

        # Shuffle row order for diversity
        row_order = list(range(len(rows)))
        rng.shuffle(row_order)

        for ri in row_order:
            if _time_left() < 1.0:
                break
            row = rows[ri]
            # Try to reduce each element (except first) from the end
            for pos in range(len(row) - 1, 0, -1):
                current_val = row[pos]
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, current_val):
                    # Remove old diffs involving this position
                    old_diffs = set()
                    for j in range(pos + 1, len(row)):
                        old_diffs.add(row[j] - row[pos])
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])

                    # Check if new_val works
                    new_diffs = set()
                    for j in range(pos + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])

                    # Check conflicts with other diffs (excluding old_diffs)
                    other_diffs = used - old_diffs
                    if new_diffs & other_diffs:
                        continue
                    # Check new_diffs don't conflict with each other
                    if len(new_diffs) != len(new_diffs):  # always true for a set
                        continue
                    # Also need to ensure row remains strictly increasing
                    if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                        continue

                    # Valid! Apply
                    used -= old_diffs
                    used |= new_diffs
                    row[pos] = new_val
                    improved = True
                    break

    return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    rng = random.Random(12345)

    # Initial construction using multiple strategies
    best_rows = None
    best_scope = float('inf')

    # Strategy 1: Pure greedy
    rows, used = _construct(n_rows, k, rng, mode='greedy')
    if rows is not None:
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)

    # Strategy 2: Randomized greedy (multiple attempts)
    for attempt in range(30):
        if _time_left() < 10.0:
            break
        rows, used = _construct(n_rows, k, rng, mode='randomized')
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Fallback: should not happen, but safety
        return [[0] * (k + 1) for _ in range(n_rows)]

    # Iterative tightening
    improved = _iterative_tightening(best_rows, k, rng, max_rounds=150)
    if _scope(improved) < best_scope:
        best_rows = improved
        best_scope = _scope(best_rows)

    # Local search
    if _time_left() > 5.0:
        ls_result = _local_search(best_rows, k, rng, max_iters=200)
        if _scope(ls_result) < best_scope:
            best_rows = ls_result
            best_scope = _scope(ls_result)

    # Another round of iterative tightening after local search
    if _time_left() > 5.0:
        improved = _iterative_tightening(best_rows, k, rng, max_rounds=100)
        if _scope(improved) < best_scope:
            best_rows = improved
            best_scope = _scope(improved)

    # Final local search
    if _time_left() > 3.0:
        ls_result = _local_search(best_rows, k, rng, max_iters=100)
        if _scope(ls_result) < best_scope:
            best_rows = ls_result
            best_scope = _scope(ls_result)

    # Validate
    if not _validate(best_rows, n_rows, k):
        # Fallback
        rows, used = _construct(n_rows, k, rng, mode='greedy')
        if rows is not None:
            best_rows = rows
        else:
            # Last resort
            best_rows = [[0] * (k + 1) for _ in range(n_rows)]

    return best_rows
