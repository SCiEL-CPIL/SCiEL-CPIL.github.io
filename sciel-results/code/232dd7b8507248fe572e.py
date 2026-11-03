import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 113.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs  # counting lower bound (scope must be at least this)
    k1 = k + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def _compute_all_diffs(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def validate_rows(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # ---- Additive-closure validation: can a difference set be a valid row? ----
    def is_valid_row_diffset(diffset, max_scope):
        """Check if diffset can be the difference set of an increasing row starting at 0.
        A valid row has k+1 elements, k*(k+1)/2 diffs, max element = max(diffset),
        and must contain 0->max as a diff. We reconstruct via backtracking on the
        row elements (not the diffs) and verify the diff set matches exactly."""
        if len(diffset) != k * (k1) // 2:
            return None
        mx = max(diffset)
        if mx > max_scope:
            return None
        if mx not in diffset:
            return None

        # Reconstruct row: elements e0=0 < e1 < ... < ek, ek = mx
        # diff set must equal {ej - ei : i<j}
        target = set(diffset)
        row = [0]

        def rec(pos, last):
            if pos == k1:
                return list(row)
            # remaining = k1 - pos elements after current last
            # last must be <= mx - (remaining) since strictly increasing
            max_x = mx - (k1 - pos - 1)
            if max_x <= last:
                return None
            # candidates: x such that {x - a for a in row} subset of target
            for x in range(last + 1, max_x + 1):
                ok = True
                for a in row:
                    if (x - a) not in target:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    res = rec(pos + 1, x)
                    if res is not None:
                        return res
                    row.pop()
            return None

        res = rec(1, 0)
        if res is None:
            return None
        # verify exact diff set match
        actual = set()
        for i in range(k1):
            for j in range(i + 1, k1):
                actual.add(res[j] - res[i])
        if actual == target:
            return res
        return None

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

    def greedy_construct(seed, strategy='basic'):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k1:
                last = row[-1]
                found = False
                if strategy == 'compact':
                    search_end = last + 25 + rng.randint(0, 15)
                elif strategy == 'spread':
                    search_end = last + 50 + rng.randint(0, 40)
                else:
                    search_end = last + 35 + rng.randint(0, 25)
                search_end = min(search_end, lb + 200)
                candidates = list(range(last + 1, search_end))
                if rng.random() < 0.5:
                    pass
                else:
                    rng.shuffle(candidates)
                for x in candidates:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    for x in range(last + 1, lb + 300):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    def backtracking_search(max_scope, seed, time_limit=3.0):
        rng = random.Random(seed)
        used_diffs = [False] * (max_scope + 1)
        rows = []
        t_start = time.time()

        def build_row(pos, last_val, row_so_far):
            nonlocal used_diffs
            if time.time() - t_start > time_limit:
                return None
            if pos == k1:
                return list(row_so_far)
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None
            valid = []
            for x in range(last_val + 1, max_x + 1):
                ok = True
                for a in row_so_far:
                    if used_diffs[x - a]:
                        ok = False
                        break
                if ok:
                    valid.append(x)
            if not valid:
                return None
            n_take = min(12, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.4:
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
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for i in range(k1):
                for j in range(i + 1, k1):
                    used_diffs[row[j] - row[i]] = True
        return rows

    # ---- NEW: global-difference-partition construction ----
    def partition_construct(max_scope, seed, time_limit=6.0):
        """Pick anchor row, then partition remaining diff pool into row diffsets,
        reconstructing each via additive-closure validation."""
        rng = random.Random(seed)
        t_start = time.time()
        diff_per_row = k * k1 // 2

        # Build anchor row within max_scope using greedy
        used = set()
        anchor = [0]
        while len(anchor) < k1:
            last = anchor[-1]
            found = False
            for x in range(last + 1, max_scope + 1):
                nd = {x - a for a in anchor}
                if not (nd & used):
                    anchor.append(x)
                    used |= nd
                    found = True
                    break
            if not found:
                return None
        if max(anchor) > max_scope:
            return None

        # Pool of remaining available differences (1..max_scope minus anchor's)
        pool = set(range(1, max_scope + 1)) - used
        if len(pool) < (n_rows - 1) * diff_per_row:
            return None

        rows = [anchor]
        for _ in range(n_rows - 1):
            if time.time() - t_start > time_limit:
                return None
            # Greedily pick diff_per_row diffs from pool that form a valid row diffset.
            # Heuristic: pick a candidate max element m, then need a set of diffs
            # including m, closed under the row structure. Instead, we do a targeted
            # backtracking over row elements restricted to pool.
            row = None
            # Try candidate max values in increasing order
            for m in sorted(pool):
                if m < k:  # need at least k distinct gaps
                    continue
                # Backtrack row elements with diffs drawn only from pool
                row = _reconstruct_from_pool(m, pool, time_limit - (time.time() - t_start))
                if row is not None:
                    break
            if row is None:
                return None
            rows.append(row)
            # Remove this row's diffs from pool
            rd = set()
            for i in range(k1):
                for j in range(i + 1, k1):
                    rd.add(row[j] - row[i])
            pool -= rd
        return rows

    def _reconstruct_from_pool(mx, pool, time_left):
        """Reconstruct a row with max element mx using only diffs in pool."""
        if time_left <= 0:
            return None
        target = set()
        row = [0]

        def rec(pos, last):
            if time_left <= 0:
                return None
            if pos == k1:
                return list(row)
            max_x = mx - (k1 - pos - 1)
            if max_x <= last:
                return None
            for x in range(last + 1, max_x + 1):
                ok = True
                for a in row:
                    d = x - a
                    if d not in pool:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    res = rec(pos + 1, x)
                    if res is not None:
                        return res
                    row.pop()
            return None

        return rec(1, 0)

    def local_improve(rows, seed, max_iters=2000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1
            for new_val in range(min_val, max_val + 1):
                valid = True
                for a in row[:elem_idx]:
                    if (new_val - a) in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        if (a - new_val) in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break
            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)
        return rows, scope

    def compact_pass(rows):
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)
        improved_any = True
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            if not improved_any:
                break
            improved_any = False
            for row_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    for a in row[:elem_pos]:
                        used_set.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used_set.discard(a - old_val)
                    row.pop(elem_pos)
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    found = False
                    for new_val in range(min_val, old_val):
                        valid = True
                        for a in row[:elem_pos]:
                            if (new_val - a) in used_set:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                if (a - new_val) in used_set:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used_set.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used_set.add(a - new_val)
                            found = True
                            break
                    if found:
                        new_scope = get_scope(rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved_any = True
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used_set.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used_set.add(a - old_val)
        return rows, scope

    # === Phase 1: quick greedy baseline ===
    base_rows = greedy_baseline()
    if base_rows:
        best_rows = [list(r) for r in base_rows]
        best_scope = get_scope(base_rows)

    # === Phase 2: backtracking at tighter scopes ===
    bt_budget = 35.0
    bt_start = time.time()
    for offset in range(0, 60):
        if time.time() - bt_start > bt_budget:
            break
        if time.time() > deadline - 35:
            break
        target = best_scope - offset
        if target < lb:
            break
        for seed in range(4):
            if time.time() - bt_start > bt_budget:
                break
            rows = backtracking_search(target, 1000 + offset * 137 + seed * 11, time_limit=2.0)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                break

    # === Phase 3: global-difference-partition constructions (NEW) ===
    part_budget = 30.0
    part_start = time.time()
    for i in range(40):
        if time.time() - part_start > part_budget:
            break
        if time.time() > deadline - 30:
            break
        target = best_scope - (i % 20)
        if target < lb:
            target = lb
        rows = partition_construct(target, 70000 + i * 173, time_limit=4.0)
        if rows is not None:
            if not validate_rows(rows):
                continue
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # === Phase 4: greedy constructions ===
    strategies = ['basic', 'compact', 'spread']
    greedy_budget = 18.0
    greedy_start = time.time()
    for i in range(2500):
        if time.time() - greedy_start > greedy_budget:
            break
        if time.time() > deadline - 30:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # === Phase 5: local improvement + compaction ===
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=1500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # === Phase 6: final backtracking tighten + polish ===
    if best_rows is not None and time.time() < deadline - 15:
        for offset in range(1, 6):
            target = best_scope - offset
            if target < lb:
                break
            if time.time() > deadline - 10:
                break
            for seed in range(3):
                if time.time() > deadline - 8:
                    break
                rows = backtracking_search(target, 50000 + offset * 1000 + seed * 73, time_limit=1.5)
                if rows is not None:
                    scope = get_scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                    break
        if time.time() < deadline - 5:
            improved, imp_scope = local_improve(best_rows, 99999, max_iters=800)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
            if time.time() < deadline - 3:
                compacted, comp_scope = compact_pass(best_rows)
                if comp_scope < best_scope:
                    best_scope = comp_scope
                    best_rows = compacted

    # Final validation
    if best_rows is not None:
        if not validate_rows(best_rows):
            fb = greedy_baseline()
            if fb:
                best_rows = fb
                best_scope = get_scope(fb)

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
