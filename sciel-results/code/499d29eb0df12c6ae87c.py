import time
import random
import math
import sys


def _row_diffs_list(row):
    """List of all positive differences in a row."""
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
    seen = set()
    for row in rows:
        for d in _row_diffs_list(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _compute_all_diffs(rows):
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """Greedy: build rows one by one, smallest valid value per position."""
    rows = []
    used_mask = 0
    high = max_scope_hint if max_scope_hint else 250

    for _ in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            for val in range(row[-1] + 1, high + 1):
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            if not found:
                # Fallback: jump far
                val = row[-1] + 20 + rng.randint(0, 30)
                row.append(val)
                for d in (val - prev for prev in row[:-1]):
                    used_mask |= (1 << d)
        rows.append(row)
    return rows


def _backtrack_row(row_prefix, used_mask, num_elements, max_val, rng, deadline):
    """
    Complete a row from a prefix, choosing values that minimize "waste"
    of scarce small differences. Uses a cost heuristic: prefer values whose
    new differences are small (they are scarcer), to keep large diffs free.
    Returns completed row or None.
    """
    if time.time() > deadline:
        return None
    if len(row_prefix) == num_elements:
        return row_prefix

    row = row_prefix
    last = row[-1]

    # Candidate values, sorted by a heuristic cost
    candidates = []
    for val in range(last + 1, max_val + 1):
        # within-row uniqueness
        new_diffs = set()
        ok = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if not ok:
            continue
        # global uniqueness
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        if new_mask & used_mask:
            continue
        # cost: prefer small max new diff (scarce) and small val
        cost = max(new_diffs) + 0.01 * val
        candidates.append((cost, val, new_mask, new_diffs))

    candidates.sort(key=lambda x: x[0])

    for cost, val, new_mask, new_diffs in candidates:
        res = _backtrack_row(row + [val], used_mask | new_mask,
                             num_elements, max_val, rng, deadline)
        if res is not None:
            return res
    return None


def _construct_rows_backtracking(n_rows, num_elements, max_scope, rng, deadline):
    """Build all rows via backtracking with a global used_mask."""
    rows = []
    used_mask = 0
    for _ in range(n_rows):
        row = [0]
        res = _backtrack_row(row, used_mask, num_elements, max_scope, rng, deadline)
        if res is None:
            return None
        rows.append(res)
        used_mask |= _row_mask(res)
    return rows


def _local_search_reduce(rows, num_elements, rng, time_limit):
    """
    Fast local search: repeatedly try to move the LAST element of a row
    (or any element) downward to reduce scope. Only accept improvements.
    """
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    cur_scope = best_scope
    gmask, rmasks = _compute_all_diffs(cur)
    n_rows = len(cur)
    n_elem = num_elements
    stagnation = 0
    max_stag = 20000

    while time.time() < time_limit and stagnation < max_stag:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = cur[row_idx]
        old = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else cur_scope
        if low >= old:
            continue
        new_val = rng.randint(low, old - 1)
        if new_val == old:
            continue

        # validity check
        row[pos] = new_val
        diffs = set()
        ok = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                break
        if ok:
            new_mask = 0
            for d in diffs:
                new_mask |= (1 << d)
            other = gmask & ~rmasks[row_idx]
            if new_mask & other:
                ok = False
        if not ok:
            row[pos] = old
            stagnation += 1
            continue

        # accept
        old_mask = rmasks[row_idx]
        row[pos] = new_val
        rmasks[row_idx] = _row_mask(row)
        gmask = (gmask & ~old_mask) | rmasks[row_idx]
        new_scope = _scope(cur)
        if new_scope < best_scope:
            best_scope = new_scope
            best = [r[:] for r in cur]
            stagnation = 0
        else:
            # revert
            row[pos] = old
            rmasks[row_idx] = old_mask
            gmask = (gmask & ~rmasks[row_idx]) | old_mask
            stagnation += 1
        cur_scope = _scope(cur)

    return best, best_scope


def _row_swap_search(rows, num_elements, rng, time_limit):
    """
    Row-unit local search: try swapping entire rows (or replacing a row with
    a freshly backtracked one at a tighter scope) to reduce max scope.
    """
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    n_rows = len(cur)
    n_elem = num_elements
    gmask, rmasks = _compute_all_diffs(cur)

    # sort rows by scope desc so we attack the biggest
    order = sorted(range(n_rows), key=lambda i: cur[i][-1], reverse=True)

    stagnation = 0
    max_stag = 30000
    while time.time() < time_limit and stagnation < max_stag:
        target_idx = order[rng.randint(0, min(3, n_rows - 1))]
        target_scope = cur[target_idx][-1]
        if target_scope <= best_scope - 1:
            stagnation += 1
            continue
        # try to rebuild this row with a tighter scope
        new_target = target_scope - 1 - rng.randint(0, 3)
        if new_target < n_elem:
            new_target = n_elem
        other_mask = gmask & ~rmasks[target_idx]
        res = _backtrack_row([0], other_mask, n_elem, new_target, rng,
                             time.time() + 5)
        if res is not None:
            new_mask = _row_mask(res)
            if not (new_mask & other_mask):
                cur[target_idx] = res
                rmasks[target_idx] = new_mask
                gmask = (gmask & ~rmasks[target_idx]) | new_mask
                new_scope = _scope(cur)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in cur]
                    stagnation = 0
                    # re-sort
                    order = sorted(range(n_rows), key=lambda i: cur[i][-1], reverse=True)
                else:
                    stagnation += 1
        else:
            stagnation += 1

    return best, best_scope


def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=30.0):
    """SA on element moves to escape local optima."""
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    cur_scope = best_scope
    gmask, rmasks = _compute_all_diffs(cur)
    n_rows = len(cur)
    n_elem = num_elements
    temp = initial_temp
    cooling = 0.9998
    min_temp = 0.5

    while time.time() < time_limit and temp > min_temp:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = cur[row_idx]
        old = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else cur_scope + 5
        if low >= high:
            temp *= cooling
            continue
        if old > low + 2 and rng.random() < 0.7:
            hi = min(old - 1, high)
            if low > hi:
                temp *= cooling
                continue
            new_val = rng.randint(low, hi)
        else:
            new_val = rng.randint(low, high)
        if new_val == old:
            temp *= cooling
            continue

        row[pos] = new_val
        diffs = set()
        ok = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                break
        if ok:
            new_mask = 0
            for d in diffs:
                new_mask |= (1 << d)
            other = gmask & ~rmasks[row_idx]
            if new_mask & other:
                ok = False
        if not ok:
            row[pos] = old
            temp *= cooling
            continue

        old_mask = rmasks[row_idx]
        rmasks[row_idx] = _row_mask(row)
        gmask = (gmask & ~old_mask) | rmasks[row_idx]
        new_scope = _scope(cur)
        delta = new_scope - cur_scope
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            cur_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [r[:] for r in cur]
        else:
            row[pos] = old
            rmasks[row_idx] = old_mask
            gmask = (gmask & ~rmasks[row_idx]) | old_mask
        temp *= cooling

    return best, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 135.0

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: CSP backtracking at progressively lower scopes
    for seed in range(6):
        if time.time() > DEADLINE - 90:
            break
        rng_local = random.Random(seed * 17 + 3)
        start_scope = min(220, counting_bound + 60)
        for scope in range(start_scope, max(start_scope - 40, counting_bound - 10), -1):
            if time.time() > DEADLINE - 85:
                break
            rows = _construct_rows_backtracking(n_rows, num_elements, scope,
                                                rng_local, DEADLINE - 80)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                break
        if best_rows is not None and best_scope <= counting_bound + 12:
            break

    # Phase 2: Greedy fallback if backtracking failed
    if best_rows is None or best_scope > counting_bound + 25:
        for seed in range(8):
            if time.time() > DEADLINE - 70:
                break
            rng_local = random.Random(seed * 11 + 5)
            rows = _greedy_construct(n_rows, num_elements, rng_local)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows

    # Fallback sequential
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 3: Row-swap search (fast, attacks max scope directly)
    if time.time() < DEADLINE - 50:
        improved, imp_scope = _row_swap_search(
            best_rows, num_elements, rng, DEADLINE - 45)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 4: Element local search
    if time.time() < DEADLINE - 35:
        improved, imp_scope = _local_search_reduce(
            best_rows, num_elements, rng, DEADLINE - 30)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 5: SA to escape
    if time.time() < DEADLINE - 25 and best_scope > counting_bound:
        improved, imp_scope = _simulated_annealing(
            best_rows, num_elements, rng, DEADLINE - 20,
            initial_temp=max(8.0, best_scope * 0.1))
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 6: Final polish
    if time.time() < DEADLINE - 10:
        improved, imp_scope = _local_search_reduce(
            best_rows, num_elements, rng, DEADLINE - 8)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
