"""Difference triangle set (7,5) solver using randomized greedy with backtracking."""
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140

    # Get a quick greedy solution to establish an upper bound
    greedy_sol = _greedy_construct(n_rows, k, deadline)
    if greedy_sol is None:
        # Fallback: return empty or raise
        return []
    best_solution = greedy_sol
    best_scope = max(max(row) for row in greedy_sol)

    # Counting bound
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    counting_bound = total_diffs

    # Try to find better solution by searching scopes from counting_bound to best_scope - 1
    for scope in range(counting_bound, best_scope):
        if time.time() > deadline - 5:
            break
        sol = _search_scope(n_rows, k, scope, deadline)
        if sol is not None:
            best_solution = sol
            best_scope = scope
            break

    return best_solution


def _greedy_construct(n_rows, k, deadline):
    """Simple greedy: for each row, pick smallest next value with no global conflict."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            found = False
            for x in range(row[-1] + 1, 100000):
                if time.time() > deadline:
                    return rows if rows else None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                return rows if rows else None
        rows.append(row)
    return rows


def _search_scope(n_rows, k, scope, deadline):
    """
    Try to find n_rows valid rows with max entry <= scope.
    Enumerate all Sidon rows in [0, scope], then backtrack-select non-conflicting ones.
    """
    # Enumerate all valid rows (Sidon sets of size k+1 in [0, scope] starting at 0)
    candidate_rows = []
    _enumerate_rows(k, scope, [0], set(), candidate_rows)

    if len(candidate_rows) < n_rows:
        return None

    # Sort by max entry ascending
    candidate_rows.sort(key=lambda x: x[1])

    # Try randomized backtracking
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
        shuffled = candidate_rows[:]
        random.shuffle(shuffled)

        selected = []
        used_diffs = set()
        if _backtrack_select(shuffled, 0, n_rows, selected, used_diffs, deadline):
            return [row for row, _ in selected]

    return None


def _enumerate_rows(remaining, scope, row, diffs, candidates):
    """
    Enumerate all strictly increasing rows of size k+1 starting at 0,
    max entry <= scope, with all internal differences distinct.
    """
    if remaining == 0:
        candidates.append((list(row), row[-1], frozenset(diffs)))
        return

    last = row[-1]
    for x in range(last + 1, scope + 1):
        new_diffs = {x - a for a in row}
        if new_diffs & diffs:
            continue
        row.append(x)
        diffs.update(new_diffs)
        _enumerate_rows(remaining - 1, scope, row, diffs, candidates)
        row.pop()
        diffs.difference_update(new_diffs)


def _backtrack_select(candidates, idx, need, selected, used_diffs, deadline):
    """
    Backtracking: select 'need' non-conflicting rows from candidates starting at idx.
    """
    if time.time() > deadline:
        return False

    if need == 0:
        return True

    if idx >= len(candidates):
        return False

    if len(candidates) - idx < need:
        return False

    for i in range(idx, len(candidates)):
        row, max_entry, diff_set = candidates[i]
        if diff_set & used_diffs:
            continue

        selected.append((row, max_entry))
        used_diffs.update(diff_set)

        if _backtrack_select(candidates, i + 1, need - 1, selected, used_diffs, deadline):
            return True

        selected.pop()
        used_diffs.difference_update(diff_set)

    return False
