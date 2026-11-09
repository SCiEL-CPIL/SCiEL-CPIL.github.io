"""Difference triangle set (4,7) – stronger construction via randomized backtracking."""
import time
import random
import sys

# We have a hard wall-clock limit; the evaluator kills at 150 s.
# Keep a safety margin so we return the best valid result well before the kill.
_DEADLINE = 150.0 - 5.0  # 145 s safety margin


def _validate(rows):
    """Return True if all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        k = len(row)
        for i in range(k):
            for j in range(i + 1, k):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k):
    """Build one row greedily: start at 0, pick smallest x whose diffs to earlier marks are unused."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
    return row


def _greedy_full(n_rows, k):
    """Full greedy construction (same as seed)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _randomized_backtrack(n_rows, k, max_scope, rng, time_limit):
    """
    Try to build n_rows rows with all entries <= max_scope.
    Uses randomized backtracking: for each row, try candidate x values in randomized order
    (biased toward small values) with occasional jumps.
    Returns a valid list of rows or None.
    """
    start_time = time.time()

    def _build_row(used, k, max_scope, rng, time_limit, start_time):
        """Build a single row via randomized backtracking."""
        row = [0]
        x = 0
        depth = 0
        while len(row) < k + 1:
            if time.time() - start_time > time_limit:
                return None
            # Try candidates from x+1 upward, with some randomization
            found = False
            # First try a small window of consecutive candidates
            window_size = 5
            for _attempt in range(window_size):
                cand = x + 1 + rng.randint(0, window_size - 1)
                if cand > max_scope:
                    break
                new_diffs = {cand - a for a in row}
                if not (new_diffs & used):
                    row.append(cand)
                    used |= new_diffs
                    x = cand
                    found = True
                    break
            if not found:
                # Try a wider random search
                for _ in range(50):
                    cand = rng.randint(x + 1, max_scope)
                    new_diffs = {cand - a for a in row}
                    if not (new_diffs & used):
                        row.append(cand)
                        used |= new_diffs
                        x = cand
                        found = True
                        break
            if not found:
                return None
            depth += 1
        return row

    used = set()
    rows = []
    for r in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        row = _build_row(used, k, max_scope, rng, time_limit, start_time)
        if row is None:
            return None
        rows.append(row)
    return rows


def _local_search_improve(rows, k, rng, time_limit, start_time):
    """
    Try to reduce the scope by swapping entries. Simple local search:
    pick a random row, pick a random entry (not 0), try to replace it with a smaller value
    that keeps all differences valid.
    """
    best_scope = _scope(rows)
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for _ in range(200):  # limited iterations per pass
            if time.time() - start_time > time_limit:
                break
            r_idx = rng.randint(0, len(rows) - 1)
            row = rows[r_idx]
            # Pick a non-zero entry to try to reduce
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            # Try smaller values
            for new_val in range(old_val - 1, 0, -1):
                # Check if replacing row[pos] with new_val keeps row strictly increasing
                if pos > 1 and new_val <= row[pos - 1]:
                    break
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    break
                # Check all differences in this row remain valid and don't conflict with other rows
                # Build the set of differences this row would have
                new_row = row[:]
                new_row[pos] = new_val
                row_diffs = set()
                valid = True
                n = len(new_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = new_row[j] - new_row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                if not valid:
                    continue
                # Check against all other rows' differences
                conflict = False
                for o_idx, o_row in enumerate(rows):
                    if o_idx == r_idx:
                        continue
                    for i in range(len(o_row)):
                        for j in range(i + 1, len(o_row)):
                            d = o_row[j] - o_row[i]
                            if d in row_diffs:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                if not conflict:
                    rows[r_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Phase 1: Greedy baseline
    rows = _greedy_full(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)

    if time.time() > deadline - 30:
        return best_rows

    # Phase 2: Try randomized constructions with various seeds and target scopes
    # Start from the greedy scope and try to do better
    greedy_scope = best_scope
    # Try a few randomized attempts with target scope = greedy_scope (to get different structures)
    for attempt in range(20):
        if time.time() > deadline - 20:
            break
        rng = random.Random(attempt * 12345 + 67890)
        target = greedy_scope
        result = _randomized_backtrack(n_rows, k, target, rng, 5.0, time.time())
        if result is not None:
            sc = _scope(result)
            if sc < best_scope:
                best_scope = sc
                best_rows = result

    if time.time() > deadline - 15:
        return best_rows

    # Phase 3: Local search on the best rows found
    if time.time() < deadline - 10:
        rng = random.Random(42)
        improved_rows, improved_scope = _local_search_improve(
            best_rows, k, rng, deadline - 10, time.time()
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Phase 4: More randomized attempts with tighter target if we improved
    if time.time() < deadline - 5:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            rng = random.Random(100000 + attempt * 999)
            target = best_scope - 1
            if target < k * (k + 1) // 2:  # too tight, skip
                break
            result = _randomized_backtrack(n_rows, k, target, rng, 3.0, time.time())
            if result is not None:
                sc = _scope(result)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = result

    return best_rows
