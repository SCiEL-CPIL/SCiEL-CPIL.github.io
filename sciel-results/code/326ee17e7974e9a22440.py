import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit

    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)

    # Total differences needed
    total_diffs = n_rows * n_diffs_per_row

    # The maximum difference value we'll ever need is bounded by scope
    # We'll use a bitmask where bit d is set if difference d is used
    # We need to track differences up to some max_scope
    # Start with a reasonable upper bound
    max_scope_initial = 200

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Try multiple constructive heuristics
    # Phase 2: Local search / simulated annealing on the best result

    # --- Phase 1: Constructive heuristics ---
    candidates = []

    # Strategy A: Simple greedy (fast baseline)
    if time.time() - start_time < deadline - 5:
        rows = _greedy_construct(n_rows, k)
        if rows:
            scope = _scope(rows)
            candidates.append((scope, rows))
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    # Strategy B: Column-wise construction with backtracking
    if time.time() - start_time < deadline - 10:
        rows = _column_wise_construct(n_rows, k, max_scope_initial)
        if rows:
            scope = _scope(rows)
            candidates.append((scope, rows))
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    # Strategy C: Randomized greedy with different orderings
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() - start_time >= deadline - 10:
            break
        random.seed(attempt * 7919 + 13)
        rows = _randomized_greedy(n_rows, k)
        if rows:
            scope = _scope(rows)
            candidates.append((scope, rows))
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    # --- Phase 2: Local search on the best candidate ---
    if best_rows is not None and time.time() - start_time < deadline - 5:
        improved_rows, improved_scope = _local_search(best_rows, best_scope, n_rows, k, deadline)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = [r[:] for r in improved_rows]

    # --- Phase 3: Simulated annealing for further improvement ---
    if best_rows is not None and time.time() - start_time < deadline - 3:
        sa_rows, sa_scope = _simulated_annealing(best_rows, best_scope, n_rows, k, deadline)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [r[:] for r in sa_rows]

    # Final validation
    if best_rows is not None:
        if _validate(best_rows, n_rows, k):
            return best_rows
        else:
            # Fallback to greedy
            return _greedy_construct(n_rows, k)

    return []


def _scope(rows):
    """Compute the scope (maximum entry) of a set of rows."""
    return max(max(row) for row in rows)


def _validate(rows, n_rows, k):
    """Check if the rows form a valid difference triangle set."""
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
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _row_diffs_bitmask(row):
    """Compute the bitmask of all positive differences in a row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask


def _row_diffs_list(row):
    """Return list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _greedy_construct(n_rows, k):
    """Simple greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = [x - a for a in row]
            if not any(d in used for d in new_diffs):
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k):
    """Greedy with random perturbations."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        attempts = 0
        while len(row) < k + 1:
            x += 1
            # Occasionally skip a value to create diversity
            if random.random() < 0.1 and x > 1:
                x += 1
            new_diffs = [x - a for a in row]
            if not any(d in used for d in new_diffs):
                row.append(x)
                used.update(new_diffs)
            attempts += 1
            if attempts > 10000:
                return None
        rows.append(row)
    return rows


def _column_wise_construct(n_rows, k, max_scope):
    """
    Column-wise construction: build all rows position by position.
    At each position, assign values to all rows simultaneously,
    choosing the smallest feasible value for each row.
    """
    # used_mask: bit d is set if difference d is used
    used_mask = 0
    rows = [[0] for _ in range(n_rows)]

    for pos in range(1, k + 1):
        # For each row, find the smallest x > rows[r][pos-1] such that
        # all new differences (x - rows[r][i] for i < pos) are not in used_mask
        # and also don't conflict with other rows being assigned at this position
        # (but since we process one row at a time within a position, we just
        # need to ensure no conflict with already-used differences)

        assigned = []
        for r in range(n_rows):
            prev = rows[r][pos - 1]
            x = prev + 1
            found = False
            while x <= max_scope:
                # Check all new differences
                conflict = False
                for i in range(pos):
                    d = x - rows[r][i]
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    # Accept this value
                    rows[r].append(x)
                    for i in range(pos):
                        d = x - rows[r][i]
                        used_mask |= (1 << d)
                    found = True
                    break
                x += 1
            if not found:
                return None

    return rows


def _local_search(rows, current_scope, n_rows, k, deadline):
    """
    Local search: try to reduce scope by moving individual elements.
    For each row, try replacing the last element with a smaller one,
    or swapping elements to reduce the maximum.
    """
    rows = [r[:] for r in rows]
    scope = current_scope
    improved = True
    no_improve_count = 0

    while time.time() < deadline - 2 and no_improve_count < 50:
        improved_this_pass = False
        for r in range(n_rows):
            if time.time() > deadline - 2:
                break
            # Try to reduce the last element of row r
            row = rows[r]
            if not row:
                continue
            last = row[-1]
            if last <= 1:
                continue

            # Compute current differences of this row
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_diffs.add(row[j] - row[i])

            # Find all used differences except this row's
            all_used = set()
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                other_row = rows[other_r]
                for i in range(len(other_row)):
                    for j in range(i + 1, len(other_row)):
                        all_used.add(other_row[j] - other_row[i])

            # Try replacing last element with smaller values
            new_last = last - 1
            while new_last > row[-2] if len(row) > 1 else 0:
                # Check if new_last creates valid differences
                new_diffs = set()
                valid = True
                for i in range(len(row) - 1):
                    d = new_last - row[i]
                    if d <= 0:
                        valid = False
                        break
                    if d in all_used:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    # Also check that new_diffs don't conflict with each other
                    # (they can't since they're all from the same row and new_last > all others)
                    # Accept the change
                    row[-1] = new_last
                    improved_this_pass = True
                    break
                new_last -= 1

            if not improved_this_pass:
                # Try moving other elements
                for pos in range(1, len(row) - 1):
                    if time.time() > deadline - 2:
                        break
                    val = row[pos]
                    if val <= 1:
                        continue
                    # Try reducing this element
                    new_val = val - 1
                    while new_val > (row[pos-1] + 1 if pos > 0 else 0):
                        # Check validity
                        new_diffs = set()
                        valid = True
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                            if d <= 0:
                                valid = False
                                break
                            if d in all_used:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            row[pos] = new_val
                            improved_this_pass = True
                            break
                        new_val -= 1
                    if improved_this_pass:
                        break

        new_scope = _scope(rows)
        if new_scope < scope:
            scope = new_scope
            no_improve_count = 0
        else:
            no_improve_count += 1

    return rows, scope


def _simulated_annealing(rows, current_scope, n_rows, k, deadline):
    """
    Simulated annealing: randomly perturb rows and accept based on temperature.
    """
    rows = [r[:] for r in rows]
    scope = current_scope
    best_rows = [r[:] for r in rows]
    best_scope = scope

    T = 10.0
    T_min = 0.01
    cooling = 0.9995

    while time.time() < deadline - 1:
        if T < T_min:
            break

        # Random perturbation: pick a random row and modify one random element
        r = random.randint(0, n_rows - 1)
        row = rows[r]
        if len(row) < 2:
            continue

        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]

        # Try to move this element
        direction = random.choice([-1, 1])
        step = random.randint(1, 5) * direction
        new_val = old_val + step

        # Ensure row stays increasing
        if pos > 0 and new_val <= row[pos - 1]:
            new_val = row[pos - 1] + 1
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            new_val = row[pos + 1] - 1

        if new_val <= 0:
            new_val = row[pos - 1] + 1 if pos > 0 else 1
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue

        # Check if this change is valid (no difference conflicts)
        # Compute differences of this row with the change
        new_row = row[:]
        new_row[pos] = new_val

        # Check all differences in new_row
        row_diffs = set()
        valid = True
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                d = new_row[j] - new_row[i]
                if d in row_diffs:
                    valid = False
                    break
                row_diffs.add(d)
        if not valid:
            continue

        # Check against all other rows
        conflict = False
        for other_r in range(n_rows):
            if other_r == r:
                continue
            other_row = rows[other_r]
            for i in range(len(other_row)):
                for j in range(i + 1, len(other_row)):
                    d = other_row[j] - other_row[i]
                    if d in row_diffs:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break

        if conflict:
            continue

        # Compute new scope
        new_scope = max(max(row2) for row2 in rows[:r] + [new_row] + rows[r+1:])
        delta = new_scope - scope

        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r] = new_row
            scope = new_scope
            if scope < best_scope:
                best_scope = scope
                best_rows = [row2[:] for row2 in rows]

        T = max(T * cooling, T_min)

    return best_rows, best_scope

# EVOLVE-BLOCK-END
