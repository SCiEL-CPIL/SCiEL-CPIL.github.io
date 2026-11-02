"""Improved constructor for the difference triangle set task."""
import time
import random
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    # Deadline guard
    deadline = time.time() + 140.0  # 10s safety margin

    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on the largest entry m:
    # We need total_diffs distinct differences, each in [1, m].
    # So m >= total_diffs.
    # Also, each row has k+1 elements, so the largest element in row i is >= k*(k+1)/2? No.
    # Actually, a simple counting bound: the number of distinct differences is at most m(m+1)/2? No.
    # The differences are in [1, m]. There are m possible values. We need total_diffs distinct values.
    # So m >= total_diffs.
    m_lb = total_diffs

    # For the specific case n_rows=4, k=7:
    # total_diffs = 4 * 7*8/2 = 4*28 = 112.
    # Best known is 134.

    # Strategy: Use a randomized local search / simulated annealing approach.
    # We'll represent the solution as a list of rows.
    # Objective: minimize the maximum element across all rows.
    # Constraint: all differences are distinct.

    # First, generate an initial feasible solution using the greedy method.
    best_rows = _greedy_construct(n_rows, k)
    best_score = _score(best_rows)

    # Now try to improve via local search
    if time.time() < deadline - 5.0:
        best_rows, best_score = _local_search(n_rows, k, best_rows, best_score, deadline)

    return best_rows


def _greedy_construct(n_rows, k):
    """Original greedy construction."""
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


def _score(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)


def _all_diffs(rows):
    """Return the set of all differences in all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    total_diffs = 0
    for row in rows:
        n = len(row)
        total_diffs += n * (n - 1) // 2
    diff_set = _all_diffs(rows)
    return len(diff_set) == total_diffs


def _local_search(n_rows, k, initial_rows, initial_score, deadline):
    """
    Local search to minimize the maximum entry.
    We'll try to reduce the largest entry by adjusting rows.
    """
    best_rows = [list(row) for row in initial_rows]
    best_score = initial_score
    current_rows = [list(row) for row in initial_rows]
    current_score = initial_score

    # Temperature for simulated annealing
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1

    while time.time() < deadline - 2.0 and temp > min_temp:
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        old_row = list(current_rows[row_idx])

        # Try to improve this row
        new_row = _improve_row(current_rows, row_idx, k)
        if new_row is None:
            # Try a random perturbation
            new_row = _perturb_row(old_row, k, current_rows, row_idx)

        if new_row is None:
            temp *= cooling
            continue

        # Check validity
        if not _is_valid_with_row(current_rows, row_idx, new_row):
            temp *= cooling
            continue

        new_score = _score_with_row(current_rows, row_idx, new_row)

        # Accept if better, or with probability exp(-delta/temp)
        if new_score < current_score:
            current_rows[row_idx] = new_row
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [list(r) for r in current_rows]
        else:
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta / temp):
                current_rows[row_idx] = new_row
                current_score = new_score

        temp *= cooling

    return best_rows, best_score


def _is_valid_with_row(rows, row_idx, new_row):
    """Check if replacing row_idx with new_row maintains validity."""
    # Compute all differences from other rows
    other_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        for j in range(len(row)):
            for l in range(j + 1, len(row)):
                other_diffs.add(row[l] - row[j])

    # Check new row's differences
    new_diffs = set()
    for j in range(len(new_row)):
        for l in range(j + 1, len(new_row)):
            d = new_row[l] - new_row[j]
            if d in other_diffs or d in new_diffs:
                return False
            new_diffs.add(d)
    return True


def _score_with_row(rows, row_idx, new_row):
    """Compute the score if row_idx is replaced with new_row."""
    max_val = max(new_row) if new_row else 0
    for i, row in enumerate(rows):
        if i != row_idx and row:
            max_val = max(max_val, max(row))
    return max_val


def _improve_row(rows, row_idx, k):
    """Try to find a better version of row row_idx."""
    # Get the used differences from other rows
    other_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        for j in range(len(row)):
            for l in range(j + 1, len(row)):
                other_diffs.add(row[l] - row[j])

    # Try greedy construction for this row, avoiding other_diffs
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & other_diffs and not new_diffs & set():
            # Also check within-row uniqueness
            if len(new_diffs) == len(row):
                row.append(x)
            else:
                continue
        # If we can't find, break
        if x > 200:
            return None
    if len(row) == k + 1:
        return row
    return None


def _perturb_row(old_row, k, rows, row_idx):
    """Create a perturbed version of old_row."""
    # Try shifting some elements
    new_row = list(old_row)
    # Randomly pick an element to adjust
    if len(new_row) > 1:
        idx = random.randint(1, len(new_row) - 1)
        delta = random.randint(-3, 3)
        new_row[idx] += delta
        # Ensure strictly increasing
        valid = True
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i - 1]:
                valid = False
                break
        if valid and new_row[0] == 0:
            return new_row
    return None

# EVOLVE-BLOCK-END
