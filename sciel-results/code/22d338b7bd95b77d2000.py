import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Total number of differences needed
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    L = k + 1  # length of each row

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper: compute max entry
    def max_entry(rows):
        return max(max(r) for r in rows)

    # Fast compaction of a single row: pull each element to its smallest feasible value
    def fast_compact_row(row, used_set):
        """In-place compaction of a row. used_set contains all diffs from OTHER rows.
        Returns True if any change was made."""
        n = len(row)
        changed = False
        for pos in range(n - 1, 0, -1):
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            # Compute old diffs for this position
            old_diffs = set()
            for j in range(n):
                if j != pos:
                    old_diffs.add(abs(row[pos] - row[j]))
            for new_val in range(lower, old_val):
                if pos < n - 1 and new_val >= row[pos + 1]:
                    break
                new_diffs = set()
                for j in range(n):
                    if j != pos:
                        new_diffs.add(abs(new_val - row[j]))
                other = used_set - old_diffs
                if new_diffs & other:
                    continue
                # Accept
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                changed = True
                break
        return changed

    def local_compact(rows, time_limit):
        """Iteratively compact all rows until no more improvement or time limit."""
        work = [list(r) for r in rows]
        used_set = set()
        for row in work:
            for d in row_diffs(row):
                used_set.add(d)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = work[r_idx]
                # Build used_set from all OTHER rows
                other_used = set()
                for i in range(n_rows):
                    if i != r_idx:
                        for d in row_diffs(work[i]):
                            other_used.add(d)
                # Also need to handle within-row constraints; use the full used_set approach
                # Recompute used_set properly
                used_set = set()
                for i in range(n_rows):
                    if i != r_idx:
                        for d in row_diffs(work[i]):
                            used_set.add(d)
                # Compact this row
                if fast_compact_row(row, used_set):
                    improved = True
        return work

    # Greedy construction
    def greedy_construct(n_rows, k, used=None, randomize=False, p_smallest=0.75, window=4, max_x=200, rng=None):
        if rng is None:
            rng = random
        if used is None:
            used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_diffs
                valid_xs = []
                while x < max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        valid_xs.append(x)
                        if len(valid_xs) >= window:
                            break
                    x += 1
                if not valid_xs:
                    return None
                if randomize:
                    if rng.random() < p_smallest:
                        next_x = valid_xs[0]
                    else:
                        next_x = rng.choice(valid_xs)
                else:
                    next_x = valid_xs[0]
                row.append(next_x)
                current_diffs |= {next_x - a for a in row[:-1]}
            rows.append(row)
            used |= current_diffs
        return rows

    # Backtracking with scope limit: try to build a row with max entry <= target
    def backtrack_row(target_scope, used, rng):
        """Try to build a row of k+1 elements starting at 0, all <= target_scope,
        with no diff conflicts with 'used'. Returns row or None."""
        row = [0]
        return _bt_recursive(row, used, target_scope, rng, 0)

    def _bt_recursive(row, used, target_scope, rng, depth):
        if time.time() > deadline:
            return None
        if len(row) == L:
            return list(row)
        # Prune: remaining elements need at least 1 each
        remaining = L - len(row)
        max_x = target_scope - (remaining - 1)
        if max_x <= row[-1]:
            return None
        # Collect candidates
        candidates = []
        for x in range(row[-1] + 1, max_x + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                candidates.append(x)
            if len(candidates) >= 40:
                break
        if not candidates:
            return None
        # Shuffle for randomness, but bias toward smaller
        if rng.random() < 0.3:
            rng.shuffle(candidates)
        else:
            # Partial shuffle: keep first few in order, shuffle rest
            if len(candidates) > 10:
                head = candidates[:10]
                tail = candidates[10:]
                rng.shuffle(tail)
                candidates = head + tail
        for x in candidates:
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used |= new_diffs
            result = _bt_recursive(row, used, target_scope, rng, depth + 1)
            if result is not None:
                return result
            used -= new_diffs
            row.pop()
        return None

    # Multi-restart strategy
    best_rows = None
    best_max = float('inf')

    # Phase 1: Greedy baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_row_diffs = set()
        for _ in range(k):
            x = row[-1] + 1
            forbidden = used | current_row_diffs
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    row.append(x)
                    current_row_diffs |= new_diffs
                    break
                x += 1
        rows.append(row)
        used |= current_row_diffs

    # Compact the greedy result
    rows = local_compact(rows, deadline - 60.0)
    best_max = max_entry(rows)
    best_rows = [r[:] for r in rows]

    # Phase 2: Multi-restart with randomized greedy + compaction
    rng = random.Random(42)
    n_restarts = 0
    while time.time() < deadline - 10.0 and n_restarts < 20:
        n_restarts += 1
        # Randomized greedy
        used_r = set()
        rows_r = None
        for attempt in range(5):
            rows_r = greedy_construct(n_rows, k, used_r, randomize=True,
                                       p_smallest=0.6 + 0.05 * attempt,
                                       window=3 + attempt, max_x=180, rng=rng)
            if rows_r is not None:
                break
        if rows_r is None:
            continue

        # Compact
        rows_r = local_compact(rows_r, deadline - 10.0)
        curr_max = max_entry(rows_r)
        if curr_max < best_max:
            best_max = curr_max
            best_rows = [r[:] for r in rows_r]

    # Phase 3: Targeted scope reduction via backtracking
    # Try to find rows with max entry = best_max - 1, best_max - 2, ...
    target = best_max - 1
    while target >= 100 and time.time() < deadline - 5.0:
        found = False
        for _ in range(3):  # Try a few times per target
            if time.time() > deadline - 5.0:
                break
            used_t = set()
            rows_t = []
            success = True
            for r_idx in range(n_rows):
                row_t = backtrack_row(target, used_t, rng)
                if row_t is None:
                    success = False
                    break
                rows_t.append(row_t)
                for d in row_diffs(row_t):
                    used_t.add(d)
            if success:
                # Verify
                all_diffs = set()
                for row_t in rows_t:
                    for d in row_diffs(row_t):
                        if d in all_diffs:
                            success = False
                            break
                        all_diffs.add(d)
                if success:
                    rows_t = local_compact(rows_t, deadline - 5.0)
                    curr_max = max_entry(rows_t)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_rows = [r[:] for r in rows_t]
                    found = True
                    break
        if found:
            target = best_max - 1  # Update target
        else:
            break

    # Phase 4: Simulated Annealing on best solution
    if time.time() < deadline - 5.0:
        rows = [r[:] for r in best_rows]
        used = set()
        for row in rows:
            for d in row_diffs(row):
                used.add(d)
        current_max = best_max

        T = 15.0
        cooling = 0.997
        min_T = 0.05

        while time.time() < deadline - 5.0:
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = set(row_diffs(old_row))
            used -= old_diffs

            # Try to regenerate with backtracking at current scope
            new_row = None
            for attempt in range(3):
                scope_try = current_max - rng.randint(0, 3)
                if scope_try < 50:
                    scope_try = 50
                new_row = backtrack_row(scope_try, used, rng)
                if new_row is not None:
                    break

            if new_row is None:
                # Fallback: randomized greedy
                new_row_list = greedy_construct(1, k, used, randomize=True,
                                                 p_smallest=0.7, window=4,
                                                 max_x=current_max, rng=rng)
                if new_row_list is not None:
                    new_row = new_row_list[0]

            if new_row is None:
                used |= old_diffs
                continue

            new_max_val = max(new_row)
            other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
            new_global_max = max(other_max, new_max_val)
            delta = new_global_max - current_max

            if delta < 0:
                rows[r_idx] = new_row
                used |= set(row_diffs(new_row))
                current_max = new_global_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used |= set(row_diffs(new_row))
                    current_max = new_global_max
                else:
                    used |= old_diffs

            T *= cooling
            if T < min_T:
                T = min_T

        # Final compaction
        if time.time() < deadline - 2.0:
            rows = local_compact(rows, deadline - 2.0)
            if max_entry(rows) < best_max:
                best_max = max_entry(rows)
                best_rows = [r[:] for r in rows]

    return best_rows
