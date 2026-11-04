"""Rewritten program: CSP + randomized local search for difference triangle set (n_rows, k)."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    for d in _row_diffs(row):
        m |= (1 << d)
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _greedy_row(max_scope, used_mask, num_elements, rng):
    """
    Greedily build one row of num_elements increasing ints starting at 0,
    with max element <= max_scope, avoiding differences in used_mask.
    Returns row or None.
    """
    # Try multiple random attempts
    for _ in range(200):
        # Pick k-1 = num_elements-2 random values from 1..max_scope, sorted
        if num_elements - 2 <= 0:
            row = [0, 1]
        else:
            vals = sorted(rng.sample(range(1, max_scope + 1), num_elements - 2))
            row = [0] + vals + [max_scope]
        # Actually we want the last element to be at most max_scope, not exactly.
        # Let me redo: pick num_elements-1 values from 1..max_scope
        if num_elements - 1 > max_scope:
            return None
        vals = sorted(rng.sample(range(1, max_scope + 1), num_elements - 1))
        row = [0] + vals
        if row[-1] > max_scope:
            continue
        # Check internal diffs are distinct
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            continue
        # Check no conflict with used_mask
        m = _row_mask(row)
        if m & used_mask:
            continue
        return row
    return None


def _greedy_construct(n_rows, num_elements, max_scope, rng, max_restarts=50):
    """
    Greedy: build rows one by one, each avoiding used differences.
    """
    for restart in range(max_restarts):
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            row = _greedy_row(max_scope, used_mask, num_elements, rng)
            if row is None:
                success = False
                break
            rows.append(row)
            used_mask |= _row_mask(row)
        if success:
            return rows
    return None


def _local_search(rows, num_elements, max_scope, rng, time_limit):
    """
    Local search: try to reduce scope by replacing individual elements.
    Returns improved rows (possibly same scope) and new scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m

    n_rows = len(current_rows)
    k = num_elements - 1  # number of non-zero elements

    def try_move_element(row_idx, pos, new_val):
        """Try setting row_idx[pos] = new_val, maintaining increasing order."""
        # pos is 1..k (index in the row, 0 is always 0)
        row = current_rows[row_idx][:]
        # Check increasing: row[pos-1] < new_val < row[pos+1]
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        row[pos] = new_val
        # Check internal diffs distinct
        diffs = _row_diffs(row)
        if len(diffs) != len(set(diffs)):
            return False
        new_mask = _row_mask(row)
        other_mask = global_mask & ~row_masks[row_idx]
        if new_mask & other_mask:
            return False
        # Also new_val must be > 0 (since pos >= 1)
        if new_val <= 0:
            return False
        # Apply
        current_rows[row_idx] = row
        row_masks[row_idx] = new_mask
        global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
        return True

    def revert_move(row_idx, old_val, pos):
        current_rows[row_idx][pos] = old_val
        row_masks[row_idx] = _row_mask(current_rows[row_idx])
        global_mask = (global_mask & ~row_masks[row_idx]) | row_masks[row_idx]

    max_iter = 20000
    for it in range(max_iter):
        if time.time() > time_limit:
            break

        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, len(current_rows[row_idx]) - 1)
        old_val = current_rows[row_idx][pos]

        # Try to reduce: sample new_val in a range below old_val
        low = current_rows[row_idx][pos - 1] + 1
        high = current_rows[row_idx][pos + 1] - 1 if pos < len(current_rows[row_idx]) - 1 else max_scope

        if low >= high:
            continue

        # Bias toward smaller values
        if rng.random() < 0.7:
            # Try to reduce
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            # Try any value in range
            new_val = rng.randint(low, high)

        if new_val == old_val:
            continue

        # Apply
        current_rows[row_idx][pos] = new_val
        row = current_rows[row_idx]
        diffs = _row_diffs(row)
        valid = len(diffs) == len(set(diffs))
        if valid:
            new_mask = _row_mask(row)
            other_mask = global_mask & ~row_masks[row_idx]
            valid = (new_mask & other_mask) == 0
        if valid:
            row_masks[row_idx] = _row_mask(row)
            global_mask = (global_mask & ~row_masks[row_idx]) | row_masks[row_idx]
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                max_scope = best_scope
        else:
            current_rows[row_idx][pos] = old_val

    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs

    # Counting bound: roughly sqrt(2 * total_diffs_needed)
    import math
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    # Best known for (7,5) is 112
    initial_scope = max(counting_bound, 100)
    max_scope_limit = 200

    rng = random.Random(42)

    # Phase 1: Greedy construction to get a valid initial solution
    best_rows = None
    best_scope = float('inf')

    for scope in range(initial_scope, max_scope_limit + 1):
        if time.time() > DEADLINE - 30:
            break
        rows = _greedy_construct(n_rows, num_elements, scope, rng, max_restarts=30)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            if s <= scope:
                # We found a solution within this scope; try to improve
                break
        if scope - initial_scope >= 15:
            break

    # If greedy failed, use a more permissive scope
    if best_rows is None:
        for scope in range(max_scope_limit, 300):
            rows = _greedy_construct(n_rows, num_elements, scope, rng, max_restarts=100)
            if rows is not None:
                best_rows = rows
                best_scope = _scope(rows)
                break
        if best_rows is None:
            # Fallback: sequential construction
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(rows)

    # Phase 2: Local search to reduce scope
    while time.time() < DEADLINE - 10:
        # Run local search
        improved_rows, improved_scope = _local_search(
            best_rows, num_elements, best_scope, rng,
            time_limit=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        # If we hit the counting bound, stop
        if best_scope <= counting_bound:
            break
        # Try a few more times with different random seeds if time permits
        if time.time() > DEADLINE - 30:
            break

    # Phase 3: Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
