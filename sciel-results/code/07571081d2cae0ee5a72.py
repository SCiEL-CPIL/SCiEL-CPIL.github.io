import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    L = k + 1  # number of elements per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # --- Fast helpers ---
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(r) for r in rows)

    def build_used(rows):
        used = set()
        for r in rows:
            used |= row_diffs(r)
        return used

    def get_diffs_for_pos(row, pos):
        diffs = set()
        for i in range(L):
            if i < pos:
                diffs.add(row[pos] - row[i])
            elif i > pos:
                diffs.add(row[i] - row[pos])
        return diffs

    def can_set(row, pos, new_val, used_set):
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        old_diffs = get_diffs_for_pos(row, pos)
        new_diffs = set()
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - row[i])
            elif i > pos:
                new_diffs.add(row[i] - new_val)
        other = used_set - old_diffs
        if new_diffs & other:
            return False
        return True

    def apply_set(row, pos, new_val, used_set):
        old_diffs = get_diffs_for_pos(row, pos)
        new_diffs = set()
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - row[i])
            elif i > pos:
                new_diffs.add(row[i] - new_val)
        for d in old_diffs:
            used_set.discard(d)
        for d in new_diffs:
            used_set.add(d)
        row[pos] = new_val

    def fast_compact_row(row, used_set):
        """Compact a single row in-place, return True if any change made."""
        changed = False
        for pos in range(L - 1, 0, -1):
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            old_diffs = get_diffs_for_pos(row, pos)
            for new_val in range(lower, old_val):
                if pos < L - 1 and new_val >= row[pos + 1]:
                    break
                new_diffs = set()
                for i in range(L):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                other = used_set - old_diffs
                if new_diffs & other:
                    continue
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                changed = True
                break
        return changed

    def local_compact(rows, time_limit):
        """Iteratively compact each element to its smallest feasible value."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = work[r_idx]
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old_val:
                        continue
                    old_diffs = get_diffs_for_pos(row, pos)
                    for new_val in range(lower, old_val):
                        if pos < L - 1 and new_val >= row[pos + 1]:
                            break
                        new_diffs = set()
                        for i in range(L):
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            elif i > pos:
                                new_diffs.add(row[i] - new_val)
                        other = used_set - old_diffs
                        if new_diffs & other:
                            continue
                        for d in old_diffs:
                            used_set.discard(d)
                        for d in new_diffs:
                            used_set.add(d)
                        row[pos] = new_val
                        improved = True
                        break
        return work

    def greedy_construct(n_rows, k, used=None, randomize=False, p_smallest=0.75, window=4, max_x=200):
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
                    if random.random() < p_smallest:
                        next_x = valid_xs[0]
                    else:
                        next_x = random.choice(valid_xs)
                else:
                    next_x = valid_xs[0]
                row.append(next_x)
                current_diffs |= {next_x - a for a in row[:-1]}
            rows.append(row)
            used |= current_diffs
        return rows

    # --- Backtracking scope search with bitset ---
    def backtracking_scope_search(target_scope, deadline, seed_offset=0):
        """Try to find a solution with max entry <= target_scope using backtracking."""
        rng = random.Random(seed_offset * 7919 + 31)
        rows = []
        used_diffs = [False] * (target_scope + 1)
        used_count = 0

        def build_row(pos, current_mark, row_so_far, depth):
            nonlocal used_count
            if time.time() > deadline:
                return None

            if pos == L:
                return list(row_so_far)

            # Prune: need at least (L - pos - 1) more values, each at least 1 apart
            max_x = target_scope - (L - pos - 1)
            if max_x <= current_mark:
                return None

            # Find valid candidates
            valid_candidates = []
            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d > target_scope or used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                if len(valid_candidates) >= 50:
                    break

            if not valid_candidates:
                return None

            # Shuffle for randomness
            rng.shuffle(valid_candidates)

            for x in valid_candidates:
                # Mark diffs
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    if used_diffs[d]:
                        # Conflict (shouldn't happen but safety)
                        break
                    new_diffs.append(d)
                else:
                    # All good, mark them
                    for d in new_diffs:
                        used_diffs[d] = True
                    used_count += len(new_diffs)

                    result = build_row(pos + 1, x, row_so_far + [x], depth + 1)
                    if result is not None:
                        return result

                    # Unmark
                    for d in new_diffs:
                        used_diffs[d] = False
                    used_count -= len(new_diffs)

            return None

        for r_idx in range(n_rows):
            if time.time() > deadline:
                return None
            row = build_row(1, 0, [0], 0)
            if row is None:
                return None
            rows.append(row)

        return rows

    # --- Initial greedy ---
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

    best_rows = [r[:] for r in rows]
    best_max = max_entry(best_rows)
    current_max = best_max

    # --- Phase 1: Local compaction ---
    if time.time() < deadline - 30:
        compacted = local_compact(rows, deadline - 30)
        if max_entry(compacted) < best_max:
            best_max = max_entry(compacted)
            best_rows = [r[:] for r in compacted]
            rows = [r[:] for r in compacted]
            current_max = best_max

    # --- Phase 2: Targeted backtracking scope reduction ---
    # Try to reduce scope by 1, 2, ...
    scope_reduced = 0
    for reduction in range(1, 15):
        target = best_max - reduction
        if target < total_diffs:  # Can't go below counting bound roughly
            break
        if time.time() > deadline - 20:
            break

        # Try a few seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            result = backtracking_scope_search(target, deadline - 20, seed_offset=seed + scope_reduced * 100)
            if result is not None:
                # Verify
                used_check = set()
                valid = True
                for r in result:
                    for d in row_diffs(r):
                        if d in used_check:
                            valid = False
                            break
                        used_check.add(d)
                    if not valid:
                        break
                if valid:
                    new_max = max_entry(result)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [r[:] for r in result]
                        rows = [r[:] for r in result]
                        current_max = best_max
                        scope_reduced = reduction
                        found = True
                        break
        if found:
            # Continue trying to reduce further
            continue
        else:
            # Couldn't reduce by this amount, stop
            break

    # --- Phase 3: Simulated Annealing ---
    T = 15.0
    cooling = 0.997
    min_T = 0.05

    while time.time() < deadline - 5:
        # Pick a random row to regenerate
        r_idx = random.randint(0, n_rows - 1)

        # Remove diffs of row r_idx from used
        old_row = rows[r_idx]
        old_diffs = set(row_diffs(old_row))
        used -= old_diffs

        # Regenerate with randomization
        new_row = None
        for attempt in range(3):
            new_row = greedy_construct(1, k, used, randomize=True,
                                       p_smallest=0.7 + 0.1 * attempt,
                                       window=3 + attempt, max_x=200)
            if new_row is not None:
                new_row = new_row[0]
                break

        if new_row is None:
            used |= old_diffs
            continue

        # Calculate new max
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)

        delta = new_global_max - current_max

        if delta < 0:
            # Accept
            rows[r_idx] = new_row
            used |= set(row_diffs(new_row))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Accept with probability
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)

            if random.random() < prob:
                rows[r_idx] = new_row
                used |= set(row_diffs(new_row))
                current_max = new_global_max
            else:
                # Rollback
                used |= old_diffs

        T *= cooling
        if T < min_T:
            T = min_T

    # --- Final compaction pass ---
    if time.time() < deadline - 2.0:
        rows = local_compact(rows, deadline - 2.0)
        if max_entry(rows) < best_max:
            best_max = max_entry(rows)
            best_rows = [r[:] for r in rows]

    # Final verification
    used_final = set()
    for r in best_rows:
        for d in row_diffs(r):
            if d in used_final:
                # Fallback to original if invalid
                return rows
            used_final.add(d)

    return best_rows
