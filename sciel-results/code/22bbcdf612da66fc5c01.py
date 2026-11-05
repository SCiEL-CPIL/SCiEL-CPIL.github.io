"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Greedy baseline
    best_rows = _greedy(n_rows, k)
    best_scope = _get_scope(best_rows)

    # Local search to improve
    best_rows, best_scope = _local_search(best_rows, n_rows, k, start_time, DEADLINE)

    # Multiple restarts with randomized greedy + local search
    attempts = 0
    while time.time() < DEADLINE - 5.0:
        attempts += 1
        # Randomized greedy: shuffle the order of adding marks with some randomness
        rows = _greedy_random(n_rows, k, seed=attempts)
        scope = _get_scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
            best_rows, best_scope = _local_search(best_rows, n_rows, k, start_time, DEADLINE)
            if time.time() > DEADLINE - 3.0:
                break

    return best_rows


def _get_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _greedy(n_rows, k):
    """Greedy ruler construction."""
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


def _greedy_random(n_rows, k, seed=0):
    """Randomized greedy with shuffled row order and perturbation."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try a few candidates with some randomness
            found = False
            for attempt in range(50):
                x += 1
                if attempt > 0:
                    # Occasionally jump ahead
                    x += rng.randint(0, 3)
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                # Fallback: keep incrementing
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
        rows.append(row)
    return rows


def _local_search(rows, n_rows, k, start_time, deadline):
    """
    Local search: try to reduce scope by adjusting entries.
    Uses difference-swap mutations and hill climbing.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(best_rows)
    current_scope = best_scope

    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Try to reduce the maximum element by modifying rows
    for iteration in range(200):
        if time.time() > deadline - 2.0:
            break

        # Find the row with the largest element
        max_idx = 0
        max_val = -1
        for i, row in enumerate(best_rows):
            if row[-1] > max_val:
                max_val = row[-1]
                max_idx = i

        # Try to reduce the last element of the max row
        target_row = list(best_rows[max_idx])
        original_last = target_row[-1]

        # Try replacing the last element with a smaller value
        for new_last in range(original_last - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
            candidate = target_row[:-1] + [new_last]
            # Check if this candidate row is internally valid
            internal_diffs = set()
            valid_internal = True
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    d = candidate[j] - candidate[i]
                    if d <= 0:
                        valid_internal = False
                        break
                    if d in internal_diffs:
                        valid_internal = False
                        break
                    internal_diffs.add(d)
            if not valid_internal:
                continue

            # Check against other rows
            other_rows = [r for i, r in enumerate(best_rows) if i != max_idx]
            other_diffs = set()
            for row in other_rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        other_diffs.add(row[j] - row[i])

            if not internal_diffs & other_diffs:
                # Valid improvement
                best_rows[max_idx] = candidate
                new_scope = _get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    current_scope = new_scope
                break
        else:
            # Try swapping differences: modify a non-last element
            # Try adjusting the second-to-last element
            if len(target_row) >= 2:
                for pos in range(len(target_row) - 2, 0, -1):
                    if time.time() > deadline - 2.0:
                        break
                    orig_val = target_row[pos]
                    for new_val in range(orig_val + 1, target_row[pos + 1]):
                        candidate = list(target_row)
                        candidate[pos] = new_val
                        # Check internal validity
                        internal_diffs = set()
                        valid_internal = True
                        for i in range(len(candidate)):
                            for j in range(i + 1, len(candidate)):
                                d = candidate[j] - candidate[i]
                                if d <= 0:
                                    valid_internal = False
                                    break
                                if d in internal_diffs:
                                    valid_internal = False
                                    break
                                internal_diffs.add(d)
                        if not valid_internal:
                            continue
                        other_rows = [r for i, r in enumerate(best_rows) if i != max_idx]
                        other_diffs = set()
                        for row in other_rows:
                            for i in range(len(row)):
                                for j in range(i + 1, len(row)):
                                    other_diffs.add(row[j] - row[i])
                        if not internal_diffs & other_diffs:
                            best_rows[max_idx] = candidate
                            new_scope = _get_scope(best_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                current_scope = new_scope
                            break
                    else:
                        continue
                    break

        # If no improvement in this iteration, try a more aggressive approach
        if current_scope == best_scope and iteration % 10 == 0:
            # Try rebuilding one row from scratch with a tighter bound
            for ri in range(n_rows):
                if time.time() > deadline - 2.0:
                    break
                other_rows = [r for i, r in enumerate(best_rows) if i != ri]
                other_diffs = set()
                for row in other_rows:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            other_diffs.add(row[j] - row[i])

                # Greedy rebuild of row ri with upper bound best_scope
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    if x > best_scope:
                        break
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & other_diffs and not (new_diffs & set()):
                        # Also check internal uniqueness
                        internal_so_far = set()
                        ok = True
                        for d in new_diffs:
                            if d in internal_so_far:
                                ok = False
                                break
                            internal_so_far.add(d)
                        if ok:
                            new_row.append(x)
                            other_diffs |= new_diffs
                if len(new_row) == k + 1:
                    best_rows[ri] = new_row
                    new_scope = _get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        current_scope = new_scope

    return best_rows, best_scope


# EVOLVE-BLOCK-END
