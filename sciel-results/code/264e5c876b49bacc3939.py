import math
import random
import time
import numpy as np

# EVOLVE-BLOCK-START


def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization as a fast baseline."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                found = False
                for _attempt in range(30):
                    step = rng.randint(1, 8)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _backtrack_row(k, used, max_val, rng, deadline):
    """
    Build a single row of k+1 elements starting at 0, all elements <= max_val,
    with all differences avoiding the global 'used' set.
    Constraint propagation: at each step, forward-check that a completion is possible.
    Returns row or None.
    """
    if time.time() > deadline:
        return None
    row = [0]

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == k + 1:
            return True

        low = row[-1] + 1
        high = max_val

        # Candidate ordering: ascending, with occasional randomization
        candidates = list(range(low, high + 1))
        if rng.random() < 0.25:
            rng.shuffle(candidates)

        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()

        return False

    if backtrack(1):
        return row
    return None


def _backtrack_construct(n_rows, k, rng, deadline, max_scope=None):
    """
    Constraint-propagation backtracking: build rows one at a time under a
    tightening max_val bound. Randomized restarts with different seeds.
    Returns the best (lowest-scope) valid set found, or None.
    """
    best_solution = None
    best_scope = max_scope if max_scope else 999999

    # Estimate an initial feasible max_val from greedy
    g = _greedy_construct(n_rows, k)
    g_scope = _total_scope(g)

    # Try tightening bounds from best-known down
    for max_val in range(g_scope, 104, -1):
        if time.time() > deadline:
            break
        # A few randomized attempts at this bound
        for attempt in range(3):
            if time.time() > deadline:
                break
            attempt_rng = random.Random(1000 * max_val + attempt + 7)
            used = set()
            rows = []
            ok = True
            for r in range(n_rows):
                if time.time() > deadline:
                    ok = False
                    break
                row = _backtrack_row(k, used, max_val, attempt_rng, deadline)
                if row is None:
                    ok = False
                    break
                rows.append(row)
                used |= _row_diffs(row)
            if ok and len(rows) == n_rows:
                scope = _total_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = [r[:] for r in rows]
                # If we hit the target bound, stop tightening
                if max_val == scope:
                    return best_solution

    return best_solution


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]

            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break

    return rows


def _simulated_annealing_scope(rows, k, rng, deadline):
    """
    Simulated annealing to escape local optima in scope minimization.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    temperature = 10.0
    cooling_rate = 0.995

    while time.time() < deadline and temperature > 0.1:
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        row = rows[idx]
        old_val = row[pos]

        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= row[pos - 1] or (pos + 1 < len(row) and new_val >= row[pos + 1]):
            continue

        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)

        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)

        remaining = used - old_diffs
        if new_diffs & remaining:
            continue

        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)

        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]

        temperature *= cooling_rate

    return best_rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    for _ in range(3):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break

    return rows


def _column_wise_construct(n_rows, k, rng, deadline, max_scope):
    """
    Column-wise construction: instead of building rows one at a time,
    build all rows column by column. At each column position, choose values
    for all rows simultaneously to minimize the maximum value while
    maintaining global difference uniqueness.
    
    This breaks the row-by-row dependency bottleneck by respecting
    global difference constraints earlier in the construction phase.
    """
    if time.time() > deadline:
        return None
    
    used = set()
    rows = [[0] for _ in range(n_rows)]
    
    for col in range(1, k + 1):
        if time.time() > deadline:
            return None
        
        # For each row, find the minimum valid value for this column
        # that doesn't conflict with used differences
        candidates = []
        for r in range(n_rows):
            prev_val = rows[r][-1]
            # Find minimum valid value for this column in this row
            val = prev_val + 1
            while True:
                new_diffs = {val - rows[r][i] for i in range(len(rows[r]))}
                if not (new_diffs & used):
                    candidates.append((val, r, new_diffs))
                    break
                val += 1
                if val > max_scope:
                    return None
        
        # Sort by value to try smaller values first
        candidates.sort(key=lambda x: x[0])
        
        # Greedily assign values, checking for conflicts
        assigned = [False] * n_rows
        for val, r, new_diffs in candidates:
            if time.time() > deadline:
                return None
            if not (new_diffs & used):
                rows[r].append(val)
                used |= new_diffs
                assigned[r] = True
        
        # Check if all rows got assigned
        if not all(assigned):
            return None
    
    return rows


def _row_pair_swap(rows, k, rng, deadline):
    """
    Row-pair swap: try to swap elements between two rows to reduce the overall scope.
    This is a structural move that can escape local optima that element-wise
    moves cannot.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    n_rows = len(rows)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Pick two rows to swap between
        r1, r2 = rng.sample(range(n_rows), 2)
        
        # Pick positions in each row
        pos1 = rng.randint(1, len(rows[r1]) - 1)
        pos2 = rng.randint(1, len(rows[r2]) - 1)
        
        val1 = rows[r1][pos1]
        val2 = rows[r2][pos2]
        
        # Try swapping
        # Check if swap is valid
        row1 = rows[r1]
        row2 = rows[r2]
        
        # Check ordering constraints
        if pos1 > 0 and val2 <= row1[pos1 - 1]:
            continue
        if pos1 + 1 < len(row1) and val2 >= row1[pos1 + 1]:
            continue
        if pos2 > 0 and val1 <= row2[pos2 - 1]:
            continue
        if pos2 + 1 < len(row2) and val1 >= row2[pos2 + 1]:
            continue
        
        # Compute old diffs
        old_diffs1 = set()
        for i in range(len(row1)):
            if i == pos1:
                continue
            if i < pos1:
                old_diffs1.add(val1 - row1[i])
            else:
                old_diffs1.add(row1[i] - val1)
        
        old_diffs2 = set()
        for i in range(len(row2)):
            if i == pos2:
                continue
            if i < pos2:
                old_diffs2.add(val2 - row2[i])
            else:
                old_diffs2.add(row2[i] - val2)
        
        # Compute new diffs
        new_diffs1 = set()
        for i in range(len(row1)):
            if i == pos1:
                continue
            if i < pos1:
                new_diffs1.add(val2 - row1[i])
            else:
                new_diffs1.add(row1[i] - val2)
        
        new_diffs2 = set()
        for i in range(len(row2)):
            if i == pos2:
                continue
            if i < pos2:
                new_diffs2.add(val1 - row2[i])
            else:
                new_diffs2.add(row2[i] - val1)
        
        remaining = used - old_diffs1 - old_diffs2
        if (new_diffs1 | new_diffs2) & remaining:
            continue
        
        # Apply swap
        row1[pos1] = val2
        row2[pos2] = val1
        used = (used - old_diffs1 - old_diffs2) | new_diffs1 | new_diffs2
        
        new_scope = _total_scope(rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in rows]
        
        # With some probability, keep the swap; otherwise revert
        if rng.random() > 0.5:
            # Revert
            row1[pos1] = val1
            row2[pos2] = val2
            used = (used - new_diffs1 - new_diffs2) | old_diffs1 | old_diffs2
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    Uses column-wise construction as primary search, then row-pair swaps and
    element-wise local search / annealing / polish as refinement.
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Phase 1: Column-wise construction (primary)
    best_rows = None
    best_scope = float('inf')
    
    # Get a baseline scope from greedy
    g = _greedy_construct(n_rows, k)
    g_scope = _total_scope(g)
    
    # Try column-wise construction with tightening bounds
    for max_val in range(g_scope, 104, -1):
        if time.time() > deadline - 40:
            break
        cw_rows = _column_wise_construct(n_rows, k, rng, deadline - 40, max_val)
        if cw_rows is not None and len(cw_rows) == n_rows:
            cw_scope = _total_scope(cw_rows)
            if cw_scope < best_scope:
                best_scope = cw_scope
                best_rows = [r[:] for r in cw_rows]
            if max_val == cw_scope:
                break
    
    # Fallback: backtracking construction
    if best_rows is None:
        bt_deadline = deadline - 30
        result = _backtrack_construct(n_rows, k, rng, bt_deadline)
        if result is not None and len(result) == n_rows:
            best_rows = [r[:] for r in result]
            best_scope = _total_scope(best_rows)
    
    # Fallback: greedy
    if best_rows is None:
        best_rows = [r[:] for r in g]
        best_scope = g_scope
    
    # Phase 2: Row-pair swaps (structural refinement)
    if time.time() < deadline - 25:
        rp_rows = _row_pair_swap(best_rows, k, rng, deadline - 25)
        rp_scope = _total_scope(rp_rows)
        if rp_scope < best_scope:
            best_rows = rp_rows
            best_scope = rp_scope
    
    # Phase 3: Local search to reduce scope
    if time.time() < deadline - 15:
        ls_rows = _local_search_scope(best_rows, k, rng, deadline - 15)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 4: Simulated annealing to escape local optima
    if time.time() < deadline - 8:
        sa_rows = _simulated_annealing_scope(best_rows, k, rng, deadline - 8)
        sa_scope = _total_scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 5: Final polish
    if time.time() < deadline - 3:
        polish_rows = _polish_pass(best_rows, k, deadline - 3)
        polish_scope = _total_scope(polish_rows)
        if polish_scope < best_scope:
            best_rows = polish_rows

    return best_rows

# EVOLVE-BLOCK-END
