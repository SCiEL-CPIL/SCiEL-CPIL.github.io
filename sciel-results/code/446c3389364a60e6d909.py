"""Difference triangle set construction with randomized greedy + local repair."""
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within each row must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s (or 150s) kill

    # Total number of differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    # Simple counting bound: need at least total_diffs distinct positive integers
    # The smallest possible max entry M must satisfy: M >= total_diffs (roughly)
    # More precisely, the differences range from 1 to M, so M >= total_diffs
    lower_bound = total_diffs

    # Best known for (7,5) is 112; counting bound is 105
    # Start search from a reasonable lower bound
    search_start = max(lower_bound, 100)

    best_result = None
    best_scope = float('inf')

    # First, try greedy to get a quick valid solution
    greedy_result = _greedy(n_rows, k)
    if greedy_result is not None:
        best_result = greedy_result
        best_scope = max(max(r) for r in greedy_result)

    # Now try to improve with randomized restarts and local search
    for attempt in range(200):
        if time.time() > deadline:
            break

        # Randomize the order of rows and use different random seeds
        random.seed(attempt * 7919 + 13)

        result = _randomized_greedy(n_rows, k, random)
        if result is not None:
            scope = max(max(r) for r in result)
            if scope < best_scope:
                best_scope = scope
                best_result = result
                # Try local improvement
                if time.time() < deadline - 20:
                    improved = _local_search(result, n_rows, k, random, deadline)
                    if improved is not None:
                        imp_scope = max(max(r) for r in improved)
                        if imp_scope < best_scope:
                            best_scope = imp_scope
                            best_result = improved

    # Try a more targeted approach: build rows one by one with backtracking
    if time.time() < deadline - 30:
        bt_result = _backtracking_greedy(n_rows, k, deadline)
        if bt_result is not None:
            scope = max(max(r) for r in bt_result)
            if scope < best_scope:
                best_scope = scope
                best_result = bt_result

    if best_result is None:
        # Fallback: return greedy result or construct a trivial valid set
        return _fallback(n_rows, k)

    return best_result


def _greedy(n_rows, k):
    """Deterministic greedy: each row takes smallest next mark with unused diffs."""
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


def _randomized_greedy(n_rows, k, rng):
    """Randomized greedy with some lookahead and perturbation."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        attempts = 0
        while len(row) < k + 1:
            attempts += 1
            if attempts > 10000:
                return None
            # Occasionally skip ahead to explore different structures
            if rng.random() < 0.1:
                x += rng.randint(1, 3)
            else:
                x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _local_search(result, n_rows, k, rng, deadline):
    """Try to reduce the scope by adjusting the last element of rows."""
    if result is None:
        return None

    rows = [list(r) for r in result]
    current_scope = max(max(r) for r in rows)

    for _ in range(500):
        if time.time() > deadline:
            break

        # Pick a random row and try to reduce its last element
        row_idx = rng.randint(0, n_rows - 1)
        row = rows[row_idx]
        old_last = row[-1]

        if old_last <= 1:
            continue

        # Try reducing the last element
        new_last = old_last - rng.randint(1, 3)
        if new_last <= row[-2]:
            continue

        # Check if new_last creates any conflict
        new_row = row[:-1] + [new_last]
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                d = new_row[j] - new_row[i]
                if d in new_diffs:
                    break
                new_diffs.add(d)
            else:
                continue
            break
        else:
            # Check against all other rows' diffs
            conflict = False
            for other_idx in range(n_rows):
                if other_idx == row_idx:
                    continue
                other_row = rows[other_idx]
                for i in range(len(other_row)):
                    for j in range(i + 1, len(other_row)):
                        d = other_row[j] - other_row[i]
                        if d in new_diffs:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break

            if not conflict:
                rows[row_idx] = new_row
                new_scope = max(max(r) for r in rows)
                if new_scope < current_scope:
                    current_scope = new_scope

    return rows


def _backtracking_greedy(n_rows, k, deadline):
    """Backtracking approach for small instances."""
    used = set()
    rows = []

    def backtrack(row_idx, current_row, x):
        if time.time() > deadline:
            return None

        if row_idx == n_rows:
            return rows[:]

        if len(current_row) == k + 1:
            rows.append(current_row[:])
            result = backtrack(row_idx + 1, [0], 0)
            if result is not None:
                return result
            rows.pop()
            return None

        # Try next positions
        for nx in range(x + 1, x + 50):
            new_diffs = {nx - a for a in current_row}
            if not new_diffs & used:
                new_row = current_row + [nx]
                new_used = used | new_diffs
                # Recurse
                saved_rows = rows[:]
                saved_used = used
                rows.append(new_row[:])
                result = backtrack(row_idx, new_row, nx)
                if result is not None:
                    return result
                rows[:] = saved_rows
                used.clear()
                used.update(saved_used)
        return None

    result = backtrack(0, [0], 0)
    return result


def _fallback(n_rows, k):
    """Construct a valid set with potentially large scope."""
    # Use a large enough range to guarantee no conflicts
    # Each row needs C(k+1,2) distinct diffs, total n_rows * C(k+1,2)
    # Use spacing to avoid conflicts
    total_diffs = n_rows * (k + 1) * k // 2
    # Use a simple construction: row i uses marks i*offset + j*offset2
    offset = total_diffs + 1
    rows = []
    for i in range(n_rows):
        row = [i * offset + j * (offset // (k + 1) + 1) for j in range(k + 1)]
        # Ensure starts at 0 for first row, but this doesn't start at 0
        # Better: use a different approach
        pass

    # Simple valid construction: use very large gaps
    base = 1000 * (k + 1)
    rows = []
    for i in range(n_rows):
        row = []
        for j in range(k + 1):
            row.append(i * base + j * (base // (k + 1) + 1))
        # Adjust to start at 0
        offset = row[0]
        row = [x - offset for x in row]
        rows.append(row)

    # Verify and fix if needed
    used = set()
    valid = True
    for row in rows:
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs or d in used:
                    valid = False
                    break
                row_diffs.add(d)
        if not valid:
            break
        used |= row_diffs

    if valid:
        return rows

    # Ultimate fallback: brute force with large range
    used = set()
    rows = []
    for i in range(n_rows):
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
