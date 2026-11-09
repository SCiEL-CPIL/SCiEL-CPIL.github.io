"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _interleaved_greedy(n_rows, k, seed=0, mode="random"):
    """
    Interleaved greedy: build all rows simultaneously.
    mode='random': shuffle row order each step.
    mode='smallest': always extend the row with the smallest current max.
    """
    rng = random.Random(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()

    for step in range(1, k + 1):
        if mode == "smallest":
            # Extend the row with smallest current last element first
            order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        else:
            order = list(range(n_rows))
            rng.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    return rows


def _greedy_row(used_diffs, max_val, rng, skip_prob=0.0):
    """Build a single row of length k+1 starting at 0, all diffs not in used_diffs, max <= max_val."""
    row = [0]
    local_diffs = set()
    while len(row) < k + 1:
        x = row[-1] + 1
        found = False
        while x <= max_val:
            if rng.random() < skip_prob and x < max_val:
                x += 1
                continue
            bad = False
            new_diffs = []
            for a in row:
                d = x - a
                if d in used_diffs or d in local_diffs:
                    bad = True
                    break
                new_diffs.append(d)
            if not bad:
                row.append(x)
                local_diffs.update(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row


def _element_local_search(rows, n_rows, k, time_limit, seed=42):
    """
    Fast element-level local search using diff_counts array.
    Systematically tries to reduce each element to the smallest valid value.
    Returns best rows found.
    """
    rng = random.Random(seed)
    start_time = time.time()
    current_rows = [list(r) for r in rows]

    def get_scope():
        return max(max(r) for r in current_rows)

    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 20
    diff_counts = [0] * (diff_size + 1)

    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1

    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()

    def try_move(r, pos, new_val):
        """Attempt to move current_rows[r][pos] to new_val. Returns True if successful."""
        row = current_rows[r]
        old_val = row[pos]
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False

        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))

        for d in old_diffs:
            diff_counts[d] -= 1

        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                pass
            if diff_counts[d] > 0:
                valid = False
                break
        if not valid:
            for d in old_diffs:
                diff_counts[d] += 1
            return False

        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
        return True

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1 if pos > 0 else 1
                found = False
                for new_val in range(lower_bound, old_val):
                    if try_move(r, pos, new_val):
                        improved = True
                        found = True
                        break
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
    return best_rows


def _row_replacement_sa(rows, n_rows, k, time_limit, seed=42):
    """
    Simulated annealing that replaces entire rows to escape local minima.
    """
    rng = random.Random(seed)
    start_time = time.time()
    current_rows = [list(r) for r in rows]

    def get_scope():
        return max(max(r) for r in current_rows)

    def get_used_diffs(rs):
        used = set()
        for r in rs:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    used = get_used_diffs(current_rows)

    T = 15.0
    T_min = 0.05
    alpha = 0.995

    while (time.time() - start_time) < time_limit and T > T_min:
        ri = rng.randint(0, n_rows - 1)
        old_row = current_rows[ri]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        used_without = used - old_diffs

        max_mark = current_scope
        new_row = None
        for attempt in range(15):
            new_row = _greedy_row(used_without, max_mark, rng, skip_prob=0.15)
            if new_row is not None:
                break

        if new_row is None:
            T *= alpha
            continue

        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])
        if new_diffs & used_without:
            T *= alpha
            continue

        other_max = 0
        for i, r in enumerate(current_rows):
            if i != ri:
                other_max = max(other_max, max(r))
        new_scope = max(other_max, max(new_row))

        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            used = used_without | new_diffs
            current_rows[ri] = new_row
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        T *= alpha
    return best_rows


def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Generate diverse initial solutions
    max_greedy_seeds = 12
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 80.0:
            break
        mode = "smallest" if seed % 2 == 0 else "random"
        rows = _interleaved_greedy(n_rows, k, seed=seed, mode=mode)
        sc = max(max(r) for r in rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        rows = []
        used = set()
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

    # Phase 2: Interleave element-level local search and row-replacement SA
    # Alternate: element LS (polish) -> row SA (escape) -> element LS -> ...
    time_remaining = deadline - time.time()
    if time_remaining > 10.0:
        # Split remaining time into chunks, alternating strategies
        # Each chunk: element LS for 40%, row SA for 60%
        # Run multiple rounds
        num_rounds = max(1, int(time_remaining // 25))
        chunk_time = time_remaining / num_rounds

        for rnd in range(num_rounds):
            if time.time() > deadline - 5.0:
                break

            # Element-level local search (polish)
            t_ls = chunk_time * 0.4
            if t_ls > 3.0:
                opt = _element_local_search(best_rows, n_rows, k, time_limit=t_ls, seed=100 + rnd)
                sc = max(max(r) for r in opt)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = opt

            # Row-replacement SA (escape)
            t_sa = deadline - time.time() - 5.0
            if t_sa > 5.0:
                opt = _row_replacement_sa(best_rows, n_rows, k, time_limit=t_sa, seed=200 + rnd)
                sc = max(max(r) for r in opt)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = opt

        # Final polish with element-level local search on the best found
        t_final = deadline - time.time() - 3.0
        if t_final > 3.0:
            opt = _element_local_search(best_rows, n_rows, k, time_limit=t_final, seed=999)
            sc = max(max(r) for r in opt)
            if sc < best_scope:
                best_scope = sc
                best_rows = opt

    return best_rows
# EVOLVE-BLOCK-END
