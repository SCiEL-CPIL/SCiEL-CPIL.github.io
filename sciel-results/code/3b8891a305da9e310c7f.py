"""Combined program: row-reconstruction ILS (inspiration 1's structural idea)
with targeted greedy descent on the global maximum and incremental diff-set
maintenance (current program's refinement machinery). The key structural change
is that local search operates at the row level (rebuilding entire rows given
the diff-set of all other rows) rather than mutating individual elements.
This provides a much richer neighborhood and avoids getting stuck in local
minima of the element-level search space."""
import time
import random

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0

    row_len = k + 1
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def greedy_init(rng, rand_prob=0.15):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                elif rng.random() < rand_prob:
                    # occasionally skip to create diversity
                    x += rng.randint(1, 3)
            rows.append(row)
        return rows

    def rebuild_row(used_other, max_val=None, rng=None, jitter=0.0):
        """Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed."""
        if rng is None:
            rng = random.Random()
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (best_scope + 40)
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and rng.random() < jitter:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def targeted_descent(rows, time_limit, rng):
        """
        Repeatedly find the global maximum element and try to reduce it
        using incremental diff-set checks. If stuck, perturb.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        cur_rows = [r[:] for r in rows]
        cur_scope = best_scope

        def compute_diff_set(rows):
            s = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        s.add(row[j] - row[i])
            return s

        cur_diff_set = compute_diff_set(cur_rows)

        while time.time() < start + time_limit:
            # Find global max
            max_val = -1
            max_r = -1
            max_c = -1
            for ri, r in enumerate(cur_rows):
                for ci, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = ri
                        max_c = ci

            if max_c == 0:
                break

            row = cur_rows[max_r]
            prev_val = row[max_c - 1]

            # Compute diffs to remove (those involving position max_c)
            remove_set = set()
            for i in range(max_c):
                remove_set.add(max_val - row[i])
            for j in range(max_c + 1, len(row)):
                remove_set.add(row[j] - max_val)

            fixed_diffs = cur_diff_set - remove_set

            found = False
            # Try to reduce max_val
            for new_val in range(max_val - 1, prev_val, -1):
                valid = True
                new_diffs = set()
                for i in range(max_c):
                    d = new_val - row[i]
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                for j in range(max_c + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue

                # Valid move
                row[max_c] = new_val
                cur_diff_set = fixed_diffs | new_diffs
                new_scope = get_scope(cur_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur_rows]
                cur_scope = new_scope
                found = True
                break

            if not found:
                # Perturb: pick a random non-max element and move it
                ri = rng.randrange(n_rows)
                ci = rng.randint(1, row_len - 1)
                if ri == max_r and ci == max_c:
                    continue
                old_val = cur_rows[ri][ci]
                lo = cur_rows[ri][ci - 1] + 1
                hi = cur_rows[ri][ci + 1] - 1 if ci < row_len - 1 else best_scope + 20
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue

                # Compute remove set for this position
                remove_set2 = set()
                r2 = cur_rows[ri]
                for i in range(ci):
                    remove_set2.add(old_val - r2[i])
                for j in range(ci + 1, len(r2)):
                    remove_set2.add(r2[j] - old_val)
                fixed2 = cur_diff_set - remove_set2

                valid2 = True
                new_diffs2 = set()
                for i in range(ci):
                    d = new_val - r2[i]
                    if d in fixed2 or d in new_diffs2:
                        valid2 = False
                        break
                    new_diffs2.add(d)
                if not valid2:
                    continue
                for j in range(ci + 1, len(r2)):
                    d = r2[j] - new_val
                    if d in fixed2 or d in new_diffs2:
                        valid2 = False
                        break
                    new_diffs2.add(d)
                if not valid2:
                    continue

                r2[ci] = new_val
                cur_diff_set = fixed2 | new_diffs2
                new_scope2 = get_scope(cur_rows)
                if new_scope2 < best_scope:
                    best_scope = new_scope2
                    best_rows = [r[:] for r in cur_rows]
                cur_scope = new_scope2

        return best_rows, best_scope

    def row_level_polish(rows, time_limit, rng):
        """
        Rebuild rows one at a time to minimize their max value,
        given the diff-set of all other rows. Process rows in order
        of decreasing max value.
        """
        start = time.time()
        cur = [list(r) for r in rows]
        cur_scope = get_scope(cur)

        # Pre-compute diff sets for all rows
        all_row_diffs = [row_diffs_set(r) for r in cur]

        while time.time() < start + time_limit:
            # Find row with largest max
            max_row_idx = max(range(n_rows), key=lambda i: max(cur[i]))
            row_max = max(cur[max_row_idx])
            if row_max <= 1:
                break

            # Compute used_other
            used_other = set()
            for j in range(n_rows):
                if j != max_row_idx:
                    used_other |= all_row_diffs[j]

            # Try to rebuild with smaller max
            best_row = cur[max_row_idx]
            best_max = row_max
            improved = False

            # Try decreasing max values
            for target_max in range(row_max - 1, max(1, row_max - 30), -1):
                new_row = rebuild_row(used_other, max_val=target_max, rng=rng, jitter=0.0)
                if new_row is not None and max(new_row) < best_max:
                    best_row = new_row
                    best_max = max(new_row)
                    improved = True
                    break

            # If greedy didn't work, try with some jitter
            if not improved:
                for attempt in range(10):
                    new_row = rebuild_row(used_other, max_val=best_max - 1, rng=rng, jitter=0.1 + 0.05 * attempt)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                        improved = True
                        break

            if improved:
                cur[max_row_idx] = best_row
                all_row_diffs[max_row_idx] = row_diffs_set(best_row)
                cur_scope = get_scope(cur)
            else:
                # Try a different row
                # Pick random row and try to improve it
                ri = rng.randrange(n_rows)
                if ri == max_row_idx:
                    ri = (ri + 1) % n_rows
                row_max2 = max(cur[ri])
                if row_max2 <= 1:
                    break

                used_other2 = set()
                for j in range(n_rows):
                    if j != ri:
                        used_other2 |= all_row_diffs[j]

                best_row2 = cur[ri]
                best_max2 = row_max2
                improved2 = False

                for target_max in range(row_max2 - 1, max(1, row_max2 - 20), -1):
                    new_row = rebuild_row(used_other2, max_val=target_max, rng=rng, jitter=0.0)
                    if new_row is not None and max(new_row) < best_max2:
                        best_row2 = new_row
                        best_max2 = max(new_row)
                        improved2 = True
                        break

                if not improved2:
                    for attempt in range(5):
                        new_row = rebuild_row(used_other2, max_val=best_max2 - 1, rng=rng, jitter=0.15 + 0.05 * attempt)
                        if new_row is not None and max(new_row) < best_max2:
                            best_row2 = new_row
                            best_max2 = max(new_row)
                            improved2 = True
                            break

                if improved2:
                    cur[ri] = best_row2
                    all_row_diffs[ri] = row_diffs_set(best_row2)
                    cur_scope = get_scope(cur)
                else:
                    break

        return cur, cur_scope

    rng = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    # Initial greedy construction
    rows = greedy_init(rng)
    scope = full_validate_scope(rows)
    if scope is not None:
        best_scope = scope
        best_rows = [r[:] for r in rows]

    # Phase 1: Row-level polish to quickly reduce scope
    if best_rows is not None and time.time() < deadline - 30:
        polished, scope_out = row_level_polish(best_rows, time_limit=20.0, rng=rng)
        if scope_out < best_scope:
            best_scope = scope_out
            best_rows = polished

    # Phase 2: Targeted descent + row-level polish iterations
    n_iterations = 60
    per_iter_time = 2.0

    for it in range(n_iterations):
        if time.time() > deadline - 5:
            break

        # Start from best, occasionally perturb
        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: rebuild 1-2 random rows with jitter
            n_perturb = rng.randint(1, 2)
            for _ in range(n_perturb):
                ri = rng.randrange(n_rows)
                used_other = set()
                for j in range(n_rows):
                    if j != ri:
                        used_other |= row_diffs_set(cur[j])
                for attempt in range(10):
                    new_row = rebuild_row(used_other, max_val=best_scope + 15, rng=rng, jitter=0.2 + 0.1 * attempt)
                    if new_row is not None:
                        cur[ri] = new_row
                        break

        scope = full_validate_scope(cur)
        if scope is None:
            continue

        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in cur]

        # Row-level polish
        if time.time() < deadline - 10:
            polished, scope_out = row_level_polish(cur, time_limit=per_iter_time, rng=rng)
            if scope_out < best_scope:
                best_scope = scope_out
                best_rows = polished

        # Targeted descent on the result
        if time.time() < deadline - 5:
            descended, scope_out = targeted_descent(best_rows, time_limit=per_iter_time, rng=rng)
            if scope_out < best_scope:
                best_scope = scope_out
                best_rows = descended

    if best_rows is None:
        best_rows = greedy_init(rng)

    return best_rows
