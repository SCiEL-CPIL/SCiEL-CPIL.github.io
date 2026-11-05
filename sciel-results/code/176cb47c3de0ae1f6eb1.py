import time
import random
import math
import sys


def _row_diffs_list(row):
    """Return list of all positive differences within a row."""
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
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _backtrack_row(row_prefix, used_mask, num_elements, max_val, rng):
    """
    Backtracking to complete a row given a prefix.
    Uses MRV-like heuristic: try candidates that leave the most remaining
    valid difference slots (scarcity heuristic).
    """
    if len(row_prefix) == num_elements:
        return row_prefix[:]

    row = row_prefix[:]
    last_val = row[-1]
    candidates = []

    for val in range(last_val + 1, max_val + 1):
        # Check within-row differences
        new_diffs = set()
        valid = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                valid = False
                break
            new_diffs.add(d)

        if not valid:
            continue

        # Check against used_mask
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)

        if new_mask & used_mask:
            continue

        # Scarcity heuristic: count how many new differences this candidate
        # would block for future rows (lower is better - leaves more room)
        # Approximate by counting bits in new_mask that are in the "tight" range
        # (small differences are more scarce)
        scarcity_score = 0
        for d in new_diffs:
            if d <= max_val // 2:
                scarcity_score += 1

        candidates.append((val, new_mask, scarcity_score))

    if not candidates:
        return None

    # Sort by scarcity score (prefer candidates that use fewer scarce differences)
    # with some randomization to avoid determinism traps
    candidates.sort(key=lambda x: (x[2], rng.random()))

    # Try candidates in order, with limited lookahead
    for val, new_mask, _ in candidates[:min(50, len(candidates))]:
        result = _backtrack_row(row + [val], used_mask | new_mask, num_elements, max_val, rng)
        if result is not None:
            return result

    return None


def _construct_rows_backtracking(n_rows, num_elements, rng, target_scope):
    """
    Construct all rows with max value <= target_scope using row-unit backtracking.
    """
    rows = []
    used_mask = 0

    for row_idx in range(n_rows):
        row = [0]
        result = _backtrack_row(row, used_mask, num_elements, target_scope, rng)
        if result is None:
            return None
        rows.append(result)
        used_mask |= _row_mask(result)

    return rows


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Greedy construction with scarcity-aware candidate selection.
    """
    rows = []
    used_mask = 0

    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            high = max_scope_hint if max_scope_hint else 200

            # Collect valid candidates
            valid_candidates = []
            for val in range(row[-1] + 1, high + 1):
                new_diffs = set()
                valid = True
                for prev in row:
                    d = val - prev
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                # Scarcity: prefer candidates using fewer small differences
                scarce = sum(1 for d in new_diffs if d <= high // 2)
                valid_candidates.append((val, new_mask, scarce))

            if not valid_candidates:
                # Fallback: just pick next available
                val = row[-1] + 1
                while True:
                    new_diffs = set(val - prev for prev in row)
                    new_mask = 0
                    for d in new_diffs:
                        new_mask |= (1 << d)
                    if not (new_mask & used_mask) and len(new_diffs) == len(set(new_diffs)):
                        break
                    val += 1
                row.append(val)
                used_mask |= _row_mask(row)
                continue

            # Sort by scarcity, pick with some randomness
            valid_candidates.sort(key=lambda x: (x[2], rng.random()))
            pick_idx = min(len(valid_candidates) - 1, rng.randint(0, min(10, len(valid_candidates) - 1)))
            val, new_mask, _ = valid_candidates[pick_idx]
            row.append(val)
            used_mask |= new_mask

        rows.append(row)

    return rows


def _row_swap_search(rows, num_elements, rng, time_limit, global_mask, row_masks):
    """
    Row-unit local search: try swapping/replacing entire rows to reduce scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    n_rows = len(current_rows)
    n_elem = num_elements

    # Precompute current global state
    global_mask, row_masks = _compute_all_diffs(current_rows)

    max_iter = 50000
    stagnation = 0
    max_stagnation = 3000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break

        # Pick a random row to try to improve
        row_idx = rng.randint(0, n_rows - 1)
        row = current_rows[row_idx]
        old_scope = _scope(current_rows)
        old_row_mask = row_masks[row_idx]

        # Try to reconstruct this row with a tighter scope
        # Target: reduce this row's max element
        target_max = row[-1] - 1
        if target_max <= 0:
            target_max = row[-1]

        # Try a few random target scopes
        for attempt in range(3):
            t = target_max - rng.randint(0, min(5, target_max // 4))
            if t <= 0:
                t = row[-1]

            # Remove this row's differences from global
            other_mask = global_mask & ~old_row_mask

            # Try to build a new row
            new_row = _backtrack_row([0], other_mask, n_elem, t, rng)
            if new_row is not None and new_row[-1] < row[-1]:
                # Check it's valid
                new_row_mask = _row_mask(new_row)
                if not (new_row_mask & other_mask):
                    # Accept
                    current_rows[row_idx] = new_row
                    row_masks[row_idx] = new_row_mask
                    global_mask = other_mask | new_row_mask

                    new_scope = _scope(current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        stagnation = 0
                    else:
                        stagnation += 1
                    break
            else:
                stagnation += 1

    return best_rows, best_scope


def _element_local_search(rows, num_elements, rng, time_limit):
    """
    Element-level local search to fine-tune after row swaps.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements

    max_iter = 100000
    stagnation = 0
    max_stagnation = 5000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break

        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]

        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope

        if low >= old_val:
            continue

        new_val = rng.randint(low, old_val - 1)

        # Try the move
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
            if not valid:
                break

        if valid:
            new_row_mask = 0
            for d in diffs_in_row:
                new_row_mask |= (1 << d)
            other_mask = global_mask & ~row_masks[row_idx]
            if not (new_row_mask & other_mask):
                # Accept
                old_row_mask = row_masks[row_idx]
                row_masks[row_idx] = new_row_mask
                global_mask = other_mask | new_row_mask
                new_scope = _scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                row[pos] = old_val
                stagnation += 1
        else:
            row[pos] = old_val
            stagnation += 1

    return best_rows, best_scope


def _simulated_annealing_rows(rows, num_elements, rng, time_limit, initial_temp=30.0):
    """
    Simulated annealing at the row-unit level: occasionally perturb a row
    by reconstructing it with a slightly different scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements

    temp = initial_temp
    cooling = 0.9997
    min_temp = 1.0

    max_iter = 100000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if temp < min_temp:
            break

        row_idx = rng.randint(0, n_rows - 1)
        row = current_rows[row_idx]
        old_row_mask = row_masks[row_idx]
        other_mask = global_mask & ~old_row_mask

        # Try to rebuild this row with a random target scope
        target = row[-1] + rng.randint(-5, 3)
        target = max(target, n_elem)  # minimum reasonable scope

        new_row = _backtrack_row([0], other_mask, n_elem, target, rng)
        if new_row is None:
            continue

        new_row_mask = _row_mask(new_row)
        if new_row_mask & other_mask:
            continue

        new_scope = max(r[-1] for r in current_rows if r is not row) if n_rows > 1 else 0
        new_scope = max(new_scope, new_row[-1])
        delta = new_scope - current_scope

        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_rows[row_idx] = new_row
            row_masks[row_idx] = new_row_mask
            global_mask = other_mask | new_row_mask
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            pass  # reject, keep current

        temp *= cooling

    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Row-unit backtracking construction at various scopes
    for seed in range(8):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 17 + 3)
        # Try scopes from upper bound downward
        start_scope = min(200, counting_bound + 40)
        found = False
        for scope in range(start_scope, max(start_scope - 25, counting_bound - 3), -1):
            rows = _construct_rows_backtracking(n_rows, num_elements, rng_local, scope)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in rows]
                found = True
                break

        if found and best_scope <= counting_bound + 8:
            break

    # Phase 2: Greedy construction as fallback/diversity
    if best_rows is None or best_scope > counting_bound + 15:
        for seed in range(8):
            if time.time() > DEADLINE - 60:
                break
            rng_local = random.Random(seed * 11 + 5)
            rows = _greedy_construct(n_rows, num_elements, rng_local)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in rows]

    # Fallback if nothing found
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    used |= new_diffs
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 3: Row-swap local search (main optimization)
    remaining = DEADLINE - time.time()
    if remaining > 50:
        improved_rows, improved_scope = _row_swap_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 45
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Phase 4: Element-level local search refinement
    if time.time() < DEADLINE - 30:
        improved_rows, improved_scope = _element_local_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 25
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Phase 5: Row-unit SA for further improvement
    if time.time() < DEADLINE - 20 and best_scope > counting_bound:
        improved_rows, improved_scope = _simulated_annealing_rows(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15,
            initial_temp=max(5.0, best_scope * 0.1)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Phase 6: Final element-level polish
    if time.time() < DEADLINE - 10:
        improved_rows, improved_scope = _element_local_search(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 8
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    used |= new_diffs
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
