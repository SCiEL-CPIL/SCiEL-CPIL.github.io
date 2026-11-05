"""Difference triangle set solver: minimize the largest entry over n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences within rows
    are distinct across all rows."""
    deadline = time.time() + 110  # safety margin before 120s limit
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # Lower bound: we need total_diffs distinct positive integers up to m
    # so m*(m+1)/2 >= total_diffs  =>  m >= approx sqrt(2*total_diffs)
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs:
        lb += 1

    # Greedy baseline
    best_rows = _greedy(n_rows, k)
    best_max = max(max(r) for r in best_rows) if best_rows else 0

    if time.time() > deadline:
        return best_rows

    # Try randomized greedy with different orderings and random choices
    random.seed(42)
    for attempt in range(200):
        if time.time() > deadline:
            break
        rows = _greedy_random(n_rows, k, seed=attempt)
        if rows:
            mx = max(max(r) for r in rows)
            if mx < best_max:
                best_max = mx
                best_rows = rows
                # If we hit the lower bound, we can stop
                if best_max <= lb:
                    break

    # Local search: try to reduce the maximum element
    if time.time() < deadline:
        best_rows, best_max = _local_search_reduce_max(best_rows, best_max, deadline, k)

    return best_rows


def _greedy(n_rows, k):
    """Deterministic greedy: pick smallest possible next element."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_random(n_rows, k, seed=0):
    """Randomized greedy: sometimes skip a candidate to explore different solutions."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        attempts = 0
        while len(row) < k + 1 and attempts < 5000:
            x += 1
            attempts += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, skip this valid choice to explore
                if rng.random() < 0.1:
                    continue
                row.append(x)
                used |= new_diffs
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows


def _local_search_reduce_max(rows, current_max, deadline, k):
    """Try to reduce the maximum element by local search."""
    best_rows = [list(r) for r in rows]
    best_max = current_max
    n_rows = len(best_rows)

    for iteration in range(50):
        if time.time() > deadline:
            break

        # Find the row with the maximum element
        max_idx = max(range(n_rows), key=lambda i: best_rows[i][-1])
        row = best_rows[max_idx]

        # Try to replace the last element with a smaller one
        old_last = row[-1]
        # Compute all diffs used by other rows
        other_used = set()
        for i in range(n_rows):
            if i == max_idx:
                continue
            r = best_rows[i]
            for a in r:
                for b in r:
                    if b > a:
                        other_used.add(b - a)

        # Try values from old_last-1 down to row[-2]+1
        found = False
        for new_last in range(old_last - 1, row[-2], -1):
            new_diffs = {new_last - a for a in row[:-1]}
            # Also check diffs within the row (excluding the last element's diffs)
            if not (new_diffs & other_used):
                # Check that new_diffs don't conflict with existing diffs in this row
                # The existing diffs in this row are those not involving the last element
                existing_diffs = set()
                for i in range(len(row) - 1):
                    for j in range(i + 1, len(row) - 1):
                        existing_diffs.add(row[j] - row[i])
                if not (new_diffs & existing_diffs):
                    row[-1] = new_last
                    found = True
                    break

        if found:
            new_max = max(max(r) for r in best_rows)
            if new_max < best_max:
                best_max = new_max
                current_max = new_max
            # Reset iteration counter to keep searching
            iteration = 0
        else:
            # Try a more aggressive approach: rebuild this row
            if time.time() > deadline:
                break
            other_used = set()
            for i in range(n_rows):
                if i == max_idx:
                    continue
                r = best_rows[i]
                for idx_a in range(len(r)):
                    for idx_b in range(idx_a + 1, len(r)):
                        other_used.add(r[idx_b] - r[idx_a])

            # Try to rebuild the row greedily but with a cap
            new_row = [0]
            x = 0
            while len(new_row) < k + 1 and x < best_max:
                x += 1
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & other_used):
                    new_row.append(x)
                    other_used |= new_diffs  # tentatively add

            if len(new_row) == k + 1:
                # Remove the tentative additions from other_used for this row
                # Actually we need to be more careful. Let's just check validity.
                best_rows[max_idx] = new_row
                # Verify full validity
                all_diffs = set()
                valid = True
                for r in best_rows:
                    row_diffs = set()
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in row_diffs or d in all_diffs:
                                valid = False
                                break
                            row_diffs.add(d)
                    if not valid:
                        break
                    all_diffs |= row_diffs
                if valid:
                    new_max = max(max(r) for r in best_rows)
                    if new_max < best_max:
                        best_max = new_max
                        current_max = new_max
                        iteration = 0
                    else:
                        # Revert
                        best_rows[max_idx] = row
                else:
                    best_rows[max_idx] = row

    return best_rows, best_max
# EVOLVE-BLOCK-END
