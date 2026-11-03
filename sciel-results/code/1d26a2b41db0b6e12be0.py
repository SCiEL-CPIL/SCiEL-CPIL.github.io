import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    k1 = k + 1  # 8 entries per row
    total_diffs = n_rows * k1 * (k1 - 1) // 2  # 4*28 = 112
    lb = total_diffs  # 112

    def get_scope(rows):
        return max(max(r) for r in rows)

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

    def backtracking_search(max_scope, seed, time_limit=4.0):
        rng = random.Random(seed)
        used_diffs = [False] * (max_scope + 1)
        rows = []
        t_start = time.time()

        def build_row(pos, last_val, row_so_far):
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

            # Try smaller values first (compact), with some randomization
            n_take = min(20, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.3:
                extra = rng.sample(valid[n_take:], min(3, len(valid) - n_take))
                window.extend(extra)
            rng.shuffle(window)

            for x in window:
                new_diffs = [x - a for a in row_so_far]
                for d in new_diffs:
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

    def compact_pass(rows):
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])

        improved_any = True
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            if not improved_any:
                break
            improved_any = False

            for row_idx in range(n_rows):
                if time.time() > deadline - 5:
                    break
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    # Remove old diffs
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

    def local_improve(rows, seed, max_iters=5000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])

        for _ in range(max_iters):
            if time.time() > deadline - 8:
                break
            row_idx = rng.randint(0, n_rows - 1)
            if max(rows[row_idx]) <= k:
                continue
            old_row = list(rows[row_idx])
            old_max = max(old_row)
            # Remove
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    used_set.discard(old_row[j] - old_row[i])
            # Try to rebuild with target slightly below old_max
            target = old_max - rng.randint(1, 8)
            if target < k:
                target = k

            def build_new(pos, last_val, rsf):
                if pos == k1:
                    return list(rsf)
                mx = min(target, scope - (k1 - pos - 1))
                if mx <= last_val:
                    return None
                valid = []
                for x in range(last_val + 1, mx + 1):
                    ok = True
                    for a in rsf:
                        if (x - a) in used_set:
                            ok = False
                            break
                    if ok:
                        valid.append(x)
                if not valid:
                    return None
                n_take = min(15, len(valid))
                window = valid[:n_take]
                if len(valid) > n_take and rng.random() < 0.4:
                    window.extend(rng.sample(valid[n_take:], min(5, len(valid) - n_take)))
                rng.shuffle(window)
                for x in window:
                    nd = [x - a for a in rsf]
                    for d in nd:
                        used_set.add(d)
                    res = build_new(pos + 1, x, rsf + [x])
                    if res is not None:
                        return res
                    for d in nd:
                        used_set.discard(d)
                return None

            new_row = build_new(1, 0, [0])
            if new_row is not None and max(new_row) < old_max:
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        used_set.add(new_row[j] - new_row[i])
                rows[row_idx] = new_row
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_set.add(old_row[j] - old_row[i])
        return rows, scope

    # Phase 1: Greedy baseline
    best_rows = greedy_baseline()
    best_scope = get_scope(best_rows)

    # Phase 2: Backtracking at various scopes
    bt_budget = 60.0
    bt_start = time.time()
    for target in range(lb, best_scope + 1):
        if time.time() - bt_start > bt_budget:
            break
        if time.time() > deadline - 50:
            break
        for seed in range(10):
            if time.time() - bt_start > bt_budget:
                break
            if time.time() > deadline - 50:
                break
            rows = backtracking_search(target, 1000 + seed * 137, time_limit=3.0)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                break

    # Phase 3: Local improvement
    if best_rows is not None and time.time() < deadline - 20:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=3000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 4: Compaction
    if best_rows is not None and time.time() < deadline - 10:
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # Phase 5: Final local improvement
    if best_rows is not None and time.time() < deadline - 5:
        improved, imp_scope = local_improve(best_rows, 99999, max_iters=2000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    if best_rows is not None and not validate_rows(best_rows):
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
