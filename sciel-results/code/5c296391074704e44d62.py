"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

_DEADLINE = None
_TIME_LIMIT = 118.0  # safety margin before 120s kill


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

    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound: smallest m such that m*(m+1)/2 >= total_diffs
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))

    # We'll use a compact representation: each row is a list of k non-zero values
    # (the full row is [0] + row_values)
    # A difference set for a row is all pairwise differences.

    def row_diffs(row_vals):
        """Return set of all positive differences in a row (which starts at 0)."""
        diffs = set()
        n = len(row_vals)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row_vals[j] - row_vals[i])
        return diffs

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

    def greedy_construct(rng, n_rows, k, bias=0.7, window=30, max_cands=15):
        """
        Greedy construction. For each row, pick values one by one.
        At each step, find valid candidates (values that don't create duplicate diffs)
        and pick with bias toward smaller values.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = []  # non-zero values
            x = 0
            while len(row) < k:
                # Find valid candidates
                valid_cands = []
                search_limit = x + window
                for candidate in range(x + 1, search_limit + 1):
                    # Check if adding candidate creates any duplicate diff
                    # Diffs with candidate: candidate - 0 = candidate, and candidate - v for v in row
                    new_diffs = {candidate} | {candidate - v for v in row}
                    if not (new_diffs & used):
                        valid_cands.append(candidate)
                        if len(valid_cands) >= max_cands:
                            break
                if not valid_cands:
                    # Extend search
                    candidate = x + 1
                    while True:
                        new_diffs = {candidate} | {candidate - v for v in row}
                        if not (new_diffs & used):
                            valid_cands.append(candidate)
                            break
                        candidate += 1

                if rng.random() < bias:
                    x = valid_cands[0]
                else:
                    x = rng.choice(valid_cands)

                row.append(x)
                used |= {x} | {x - v for v in row[:-1]}
            rows.append(row)
        return rows

    def local_search(rows, k, max_iters=300):
        """
        Try to reduce the maximum entry by adjusting individual row values.
        rows: list of lists of k non-zero values
        Returns improved rows or None.
        """
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            used |= full_row_diffs(row)

        improved = False
        for _ in range(max_iters):
            if _time_left() < 3.0:
                break
            # Find the row with the largest last entry
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            max_val = row[-1]

            # Try to reduce the last element of this row
            found = False
            for new_val in range(row[-2] + 1, max_val):
                # Check if replacing last element with new_val keeps all diffs valid
                old_diffs = full_row_diffs(row)
                new_row = row[:-1] + [new_val]
                new_diffs = full_row_diffs(new_row)
                # Diffs that are in new_diffs but not in old_diffs must not conflict with used - old_diffs
                added_diffs = new_diffs - old_diffs
                if added_diffs & (used - old_diffs):
                    continue
                # Also check that new_val > row[-2]
                if new_val <= row[-2]:
                    continue
                # Valid! Apply
                used -= old_diffs
                used |= new_diffs
                row[-1] = new_val
                found = True
                improved = True
                break

            if not found:
                # Try other rows
                found_any = False
                for idx in range(len(rows)):
                    if idx == max_idx:
                        continue
                    row = rows[idx]
                    # Try to reduce the last element
                    max_val = row[-1]
                    for new_val in range(row[-2] + 1, max_val):
                        old_diffs = full_row_diffs(row)
                        new_row = row[:-1] + [new_val]
                        new_diffs = full_row_diffs(new_row)
                        added_diffs = new_diffs - old_diffs
                        if added_diffs & (used - old_diffs):
                            continue
                        if new_val <= row[-2]:
                            continue
                        used -= old_diffs
                        used |= new_diffs
                        row[-1] = new_val
                        found_any = True
                        improved = True
                        break
                    if found_any:
                        break
                if not found_any:
                    break

        if improved:
            return rows
        return None

    def validate(rows, n_rows, k):
        """Validate that all differences are distinct across all rows."""
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k:
                return False
            # Check strictly increasing (and positive)
            prev = 0
            for v in row:
                if v <= prev:
                    return False
                prev = v
            # Check diffs
            diffs = full_row_diffs(row)
            if diffs & used:
                return False
            used |= diffs
        return True

    # Run multiple randomized attempts
    rng = random.Random(42)

    # First, try deterministic greedy to get a baseline
    best_rows = greedy_construct(rng, n_rows, k, bias=1.0)
    best_max = max(max(r) for r in best_rows)

    # Try various randomized greedy configurations
    num_attempts = 80
    for attempt in range(num_attempts):
        if _time_left() < 5.0:
            break
        bias = 0.5 + 0.4 * rng.random()
        window = 20 + int(30 * rng.random())
        max_cands = 10 + int(20 * rng.random())
        rows = greedy_construct(rng, n_rows, k, bias=bias, window=window, max_cands=max_cands)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            # Try local search on the best
            if _time_left() > 5.0:
                improved = local_search(best_rows, k, max_iters=150)
                if improved:
                    new_max = max(max(r) for r in improved)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = improved

    # Final local search on best
    if _time_left() > 3.0:
        improved = local_search(best_rows, k, max_iters=500)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # Convert from compact representation (k non-zero values) to full rows (k+1 values starting at 0)
    full_rows = [[0] + row for row in best_rows]

    # Validate the result
    if not validate(full_rows, n_rows, k):
        # Fallback to deterministic greedy
        fallback = greedy_construct(rng, n_rows, k, bias=1.0)
        full_rows = [[0] + row for row in fallback]

    return full_rows
