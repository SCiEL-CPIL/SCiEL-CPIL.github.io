"""Difference triangle set (7,5): CSP init + efficient simulated annealing."""
import time
import random

def _validate(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(r) for r in rows)

def _greedy_init(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _csp_init(n_rows, k, max_scope, time_limit):
    start = time.time()
    used = set()
    rows = []

    def backtrack(row_idx, row, x):
        nonlocal used, rows
        if time.time() - start > time_limit:
            return False
        if len(row) == k + 1:
            rows.append(list(row))
            return True
        for trial in range(x + 1, max_scope + 1):
            new_diffs = {trial - a for a in row}
            if new_diffs & used:
                continue
            remaining = (k + 1) - len(row) - 1
            if remaining > 0 and max_scope - trial < remaining:
                continue
            used |= new_diffs
            row.append(trial)
            if row_idx + 1 < n_rows:
                if backtrack(row_idx + 1, [], 0):
                    return True
            else:
                return True
            row.pop()
            used -= new_diffs
        return False

    if backtrack(0, [0], 0):
        return rows
    return None

def _sa_optimize(rows, n_rows, k, time_limit):
    start = time.time()
    if not _validate(rows):
        return rows

    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]

    # Precompute per-row diff sets and global used
    def row_diff_set(row):
        s = set()
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                s.add(row[j] - ai)
        return s

    row_diffs = [row_diff_set(r) for r in current_rows]
    global_used = set()
    for rd in row_diffs:
        global_used |= rd

    T = 50.0
    T_min = 0.5
    cooling = 0.999

    steps = 0
    while T > T_min and (time.time() - start) < time_limit:
        r_idx = random.randrange(n_rows)
        pos = random.randrange(1, k + 1)
        old_val = current_rows[r_idx][pos]

        # Generate candidate
        if random.random() < 0.6:
            new_val = random.randint(1, max(1, old_val))
        else:
            new_val = random.randint(1, old_val + 15)

        if new_val == old_val:
            continue

        new_row = list(current_rows[r_idx])
        new_row[pos] = new_val
        if pos > 0 and new_row[pos] <= new_row[pos - 1]:
            continue
        if pos < k and new_row[pos] >= new_row[pos + 1]:
            continue

        # Compute new diff set for this row
        new_rd = row_diff_set(new_row)

        # Check collision with other rows
        other_used = global_used - row_diffs[r_idx]
        if new_rd & other_used:
            continue

        # Compute scope delta
        old_max = max(r[-1] for r in current_rows)
        new_max = max([r[-1] for i, r in enumerate(current_rows) if i != r_idx] + [new_row[-1]])
        delta = new_max - old_max

        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            global_used -= row_diffs[r_idx]
            global_used |= new_rd
            row_diffs[r_idx] = new_rd
            current_rows[r_idx] = new_row
            steps += 1
            if new_max < best_scope:
                best_scope = new_max
                best_rows = [list(r) for r in current_rows]
        T *= cooling

    return best_rows

def construct(n_rows, k):
    random.seed(12345)
    deadline = time.time() + 135.0  # 15s safety margin

    # Phase 1: Try CSP init with various scope bounds
    c = (k + 1) * k // 2
    base = n_rows * c
    rows = None
    for margin in [0, 3, 5, 10, 15, 25, 40]:
        ms = base + margin
        if time.time() > deadline - 25.0:
            break
        res = _csp_init(n_rows, k, ms, time_limit=4.0)
        if res is not None and _validate(res):
            rows = res
            break

    if rows is None:
        rows = _greedy_init(n_rows, k)

    # Phase 2: SA polish
    if time.time() < deadline - 5.0:
        sa_time = min(80.0, deadline - 5.0 - time.time())
        if sa_time > 1.0:
            rows = _sa_optimize(rows, n_rows, k, sa_time)

    if not _validate(rows):
        rows = _greedy_init(n_rows, k)

    return rows
