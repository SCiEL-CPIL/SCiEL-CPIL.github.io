import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Total number of positive differences needed
    num_diffs = n_rows * k * (k + 1) // 2
    lb = num_diffs  # counting lower bound (scope must be at least this)

    best_rows = None
    best_scope = float('inf')
    k1 = k + 1

    def time_ok(margin=3.0):
        return time.time() < deadline - margin

    # --- Fast greedy baseline (smallest valid x each step) ---
    def greedy_baseline():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def get_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def scope_of(rows):
        return max(max(r) for r in rows)

    # --- Row-level local search: rebuild a row to reduce its max ---
    def rebuild_row(row, used, max_scope, rng, tries=30):
        """Try to rebuild a row (same length) to have a smaller max,
        keeping all differences disjoint from used (minus this row's own)."""
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])
        other_used = used - old_diffs
        row_len = len(row)
        target_max = row[-1] - 1
        for _ in range(tries):
            # Random restart: build greedily but with some randomization
            new_row = [0]
            x = 0
            success = True
            while len(new_row) < row_len:
                x += 1
                if x > target_max:
                    success = False
                    break
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & other_used):
                    new_row.append(x)
                else:
                    # Try a random jump
                    x = rng.randint(1, target_max)
            if success and len(new_row) == row_len:
                return new_row
        return None

    def local_improve_rowlevel(rows, seed, max_iters=500):
        """Row-level local search: try to rebuild each row to reduce scope."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = scope_of(rows)
        used = get_used(rows)

        for _ in range(max_iters):
            if not time_ok(5.0):
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if row[-1] <= lb // n_rows + 1:
                continue

            new_row = rebuild_row(row, used, row[-1] - 1, rng, tries=20)
            if new_row is not None and new_row[-1] < row[-1]:
                # Update used set
                old_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        old_diffs.add(row[j] - row[i])
                used -= old_diffs
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                used |= new_diffs
                rows[row_idx] = new_row
                new_scope = scope_of(rows)
                if new_scope < scope:
                    scope = new_scope

        return rows, scope

    # --- Single-element shift local improvement ---
    def local_improve(rows, seed, max_iters=2000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = scope_of(rows)

        used = get_used(rows)

        for _ in range(max_iters):
            if not time_ok(3.0):
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove element and its diffs
            for a in row[:elem_idx]:
                used.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used.discard(a - old_val)
            row.pop(elem_idx)

            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1
            found = False
            for new_val in range(min_val, max_val + 1):
                valid = True
                for a in row[:elem_idx]:
                    if (new_val - a) in used:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        if (a - new_val) in used:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = scope_of(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used.add(a - old_val)

        return rows, scope

    # --- Systematic compaction pass ---
    def compact_pass(rows, seed=0):
        rows = [list(r) for r in rows]
        scope = scope_of(rows)
        used = get_used(rows)

        improved = True
        rounds = 0
        while improved and time_ok(2.0) and rounds < 200:
            improved = False
            rounds += 1
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    for a in row[:elem_pos]:
                        used.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used.discard(a - old_val)
                    row.pop(elem_pos)
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    found = False
                    for new_val in range(min_val, old_val):
                        valid = True
                        for a in row[:elem_pos]:
                            if (new_val - a) in used:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                if (a - new_val) in used:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used.add(a - new_val)
                            found = True
                            break
                    if found:
                        new_scope = scope_of(rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved = True
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used.add(a - old_val)
        return rows, scope

    # --- Backtracking search with MRV for tighter solutions ---
    def backtracking_search(max_scope, seed, time_limit=2.0):
        rng = random.Random(seed)
        t_start = time.time()
        used_diffs = [False] * (max_scope + 1)
        rows = []

        def build_row(pos, last_val, row_so_far):
            if time.time() - t_start > time_limit:
                return None
            if pos == k1:
                return list(row_so_far)

            # Prune: need (k1 - pos) more elements
            remaining = k1 - pos
            max_x = max_scope - (remaining - 1)
            if max_x <= last_val:
                return None

            # MRV: count valid choices for each position to pick the most constrained
            # For speed, just pick the next element greedily with randomization
            valid = []
            for x in range(last_val + 1, min(max_x + 1, last_val + 200)):
                ok = True
                for a in row_so_far:
                    if used_diffs[x - a]:
                        ok = False
                        break
                if ok:
                    valid.append(x)

            if not valid:
                return None

            # Bias toward smaller values, shuffle a window
            n_take = min(15, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.3:
                extra = rng.sample(valid[n_take:], min(5, len(valid) - n_take))
                window.extend(extra)
            rng.shuffle(window)

            for x in window:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                for d in new_diffs:
                    used_diffs[d] = False
            return None

        for _ in range(n_rows):
            if time.time() - t_start > time_limit:
                return None
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for i in range(k1):
                for j in range(i + 1, k1):
                    used_diffs[row[j] - row[i]] = True
        return rows

    # --- Backtracking with MRV: choose the most constrained position first ---
    def backtracking_mrv(max_scope, seed, time_limit=3.0):
        """Backtracking that uses Minimum Remaining Values heuristic:
        at each step, choose the variable (position) with fewest valid values."""
        rng = random.Random(seed)
        t_start = time.time()
        used = set()
        rows = [[] for _ in range(n_rows)]

        def is_valid_val(row, pos, val, other_rows):
            """Check if val can go at position pos in row without conflicts."""
            # Must be > previous element
            if pos > 0 and val <= row[pos - 1]:
                return False
            # Must leave room for remaining elements
            remaining = k1 - pos - 1
            if val + remaining > max_scope:
                return False
            # Check differences
            for i in range(pos):
                d = val - row[i]
                if d in used:
                    return False
            # Check against other rows
            for orow in other_rows:
                for i in range(len(orow)):
                    d = abs(val - orow[i])
                    if d > 0 and d in used:
                        return False
            return True

        def count_valid(row, pos, max_x, other_rows):
            """Count valid values for position pos in row."""
            count = 0
            min_x = row[pos - 1] + 1 if pos > 0 else 1
            for x in range(min_x, min(max_x + 1, min_x + 100)):
                if is_valid_val(row, pos, x, other_rows):
                    count += 1
                    if count >= 100:
                        break
            return count

        def solve():
            if time.time() - t_start > time_limit:
                return False
            # Find the most constrained unfilled position
            best_pos = None
            best_row_idx = None
            min_count = float('inf')

            for ri in range(n_rows):
                for pos in range(k1):
                    if pos < len(rows[ri]):
                        continue
                    if pos > 0 and pos > len(rows[ri]):
                        break  # can't skip positions
                    # Only consider the next unfilled position in each row
                    if pos == len(rows[ri]):
                        max_x = max_scope - (k1 - pos - 1)
                        min_x = rows[ri][pos - 1] + 1 if pos > 0 else 1
                        if min_x > max_x:
                            return False
                        cnt = count_valid(rows[ri], pos, max_x, [rows[oj] for oj in range(n_rows) if oj != ri])
                        if cnt == 0:
                            return False
                        if cnt < min_count:
                            min_count = cnt
                            best_pos = pos
                            best_row_idx = ri
                            if cnt == 1:
                                break
                if best_pos is not None and min_count == 1:
                    break

            if best_pos is None:
                return True  # all filled

            ri = best_row_idx
            pos = best_pos
            row = rows[ri]
            max_x = max_scope - (k1 - pos - 1)
            min_x = row[pos - 1] + 1 if pos > 0 else 1

            # Get all valid values
            valid_vals = []
            for x in range(min_x, max_x + 1):
                if is_valid_val(row, pos, x, [rows[oj] for oj in range(n_rows) if oj != ri]):
                    valid_vals.append(x)

            if not valid_vals:
                return False

            # Shuffle for randomness, bias toward smaller
            if len(valid_vals) > 20:
                window = valid_vals[:15]
                if rng.random() < 0.3:
                    window.extend(rng.sample(valid_vals[15:], min(5, len(valid_vals) - 15)))
                rng.shuffle(window)
                valid_vals = window

            for x in valid_vals:
                # Assign
                row.append(x)
                new_diffs = set()
                for i in range(pos):
                    d = x - row[i]
                    new_diffs.add(d)
                    used.add(d)

                if solve():
                    return True

                # Unassign
                row.pop()
                for d in new_diffs:
                    used.discard(d)
            return False

        if solve():
            return rows
        return None

    # --- Phase 1: Greedy baseline ---
    rows = greedy_baseline()
    if rows:
        best_scope = scope_of(rows)
        best_rows = [list(r) for r in rows]

    # --- Phase 2: Row-level local improvement ---
    if best_rows and time_ok(10.0):
        improved, imp_scope = local_improve_rowlevel(best_rows, 42, max_iters=300)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # --- Phase 3: Single-element local improvement ---
    if best_rows and time_ok(10.0):
        improved, imp_scope = local_improve(best_rows, 42, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # --- Phase 4: Backtracking with decreasing target scope ---
    target = best_scope - 1
    attempt = 0
    while target > lb and time_ok(15.0):
        found = False
        for _ in range(15):
            if time.time() > deadline - 15:
                break
            sol = backtracking_search(target, 100000 + attempt, time_limit=2.0)
            if sol:
                new_scope = scope_of(sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in sol]
                    found = True
                    # Improve
                    if time_ok(15.0):
                        improved, imp_scope = local_improve(best_rows, attempt + 500, max_iters=300)
                        if imp_scope < best_scope:
                            best_scope = imp_scope
                            best_rows = improved
                        compacted, comp_scope = compact_pass(best_rows)
                        if comp_scope < best_scope:
                            best_scope = comp_scope
                            best_rows = compacted
                    break
            attempt += 1
        if found:
            target = best_scope - 1
        else:
            target -= 1
            if target <= lb:
                break

    # --- Phase 5: MRV backtracking for tighter solutions ---
    if time_ok(20.0):
        target_mrv = best_scope - 1
        attempt_mrv = 0
        while target_mrv > lb and time_ok(10.0):
            found_mrv = False
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                sol = backtracking_mrv(target_mrv, 200000 + attempt_mrv, time_limit=3.0)
                if sol:
                    new_scope = scope_of(sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in sol]
                        found_mrv = True
                        if time_ok(10.0):
                            improved, imp_scope = local_improve_rowlevel(best_rows, attempt_mrv + 1000, max_iters=200)
                            if imp_scope < best_scope:
                                best_scope = imp_scope
                                best_rows = improved
                            improved2, imp_scope2 = local_improve(best_rows, attempt_mrv + 2000, max_iters=300)
                            if imp_scope2 < best_scope:
                                best_scope = imp_scope2
                                best_rows = improved2
                            compacted, comp_scope = compact_pass(best_rows)
                            if comp_scope < best_scope:
                                best_scope = comp_scope
                                best_rows = compacted
                        break
                attempt_mrv += 1
            if found_mrv:
                target_mrv = best_scope - 1
            else:
                target_mrv -= 1
                if target_mrv <= lb:
                    break

    # --- Phase 6: More local improvement and compaction ---
    if best_rows and time_ok(10.0):
        improved, imp_scope = local_improve_rowlevel(best_rows, 99999, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        improved2, imp_scope2 = local_improve(best_rows, 99998, max_iters=1000)
        if imp_scope2 < best_scope:
            best_scope = imp_scope2
            best_rows = improved2
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # --- Phase 7: Final random restarts if time remains ---
    if best_rows and time_ok(10.0):
        for i in range(20):
            if time.time() > deadline - 5:
                break
            sol = backtracking_search(best_scope - 1, 500000 + i * 131, time_limit=1.5)
            if sol:
                new_scope = scope_of(sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in sol]
                    if time_ok(5.0):
                        improved, imp_scope = local_improve(best_rows, 700000 + i, max_iters=200)
                        if imp_scope < best_scope:
                            best_scope = imp_scope
                            best_rows = improved

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
