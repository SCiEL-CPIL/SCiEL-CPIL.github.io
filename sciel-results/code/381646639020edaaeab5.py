import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 145.0
    row_len = k + 1
    num_diffs_per_row = row_len * (row_len - 1) // 2  # 15 for k=5

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def greedy_row(used_mask):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            new_mask = 0
            for a in row:
                new_mask |= (1 << (x - a))
            if (new_mask & used_mask) == 0:
                row.append(x)
                used_mask |= new_mask
        return row

    def greedy_row_capped(used_mask, cap):
        row = [0]
        x = 0
        local_mask = 0
        while len(row) < row_len:
            found = False
            for t in range(x + 1, cap + 1):
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (t - a))
                if (new_mask & used_mask) or (new_mask & local_mask):
                    continue
                row.append(t)
                local_mask |= new_mask
                x = t
                found = True
                break
            if not found:
                return None
        return row

    def row_from_diffs(diffs):
        row = [0]
        for d in diffs:
            row.append(row[-1] + d)
        return row

    def diffs_from_row(row):
        return [row[i + 1] - row[i] for i in range(len(row) - 1)]

    def pairwise_diffs(row):
        L = len(row)
        diffs = []
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.append(row[j] - ri)
        return diffs

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    def build_diff_counts(rows):
        counts = Counter()
        for r in rows:
            for d in pairwise_diffs(r):
                counts[d] = counts.get(d, 0) + 1
        return counts

    def remove_row_diffs(dc, row):
        for d in pairwise_diffs(row):
            dc[d] -= 1
            if dc[d] == 0:
                del dc[d]

    def add_row_diffs(dc, row):
        for d in pairwise_diffs(row):
            dc[d] = dc.get(d, 0) + 1

    def find_row_dfs(used_mask, best_max_init, S_cap, node_budget, dl):
        best = [None]
        best_max = [best_max_init]
        nodes = [0]

        def dfs(row, rmask, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = row[:]
                return
            hi = min(best_max[0] - 1, S_cap)
            lo = last + 1
            if lo > hi:
                return
            for x in range(lo, hi + 1):
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (x - a))
                if (new_mask & used_mask) or (new_mask & rmask):
                    continue
                row.append(x)
                dfs(row, rmask | new_mask, x)
                row.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return

        dfs([0], 0, 0)
        if best[0] is None:
            return None
        return best[0], row_mask(best[0])

    def sa_optimize(rows, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_diffs = [diffs_from_row(r) for r in cur_rows]
        cur_scope = scope_of(cur_rows)
        cur_sum = sum_entries(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_diffs = [d[:] for d in cur_diffs]
        best_scope = cur_scope
        best_sum = cur_sum

        diff_counts = build_diff_counts(cur_rows)
        row_masks = [row_mask(r) for r in cur_rows]
        total_mask = 0
        for rm in row_masks:
            total_mask |= rm

        def get_cost(rows, dc, s):
            collisions = sum(max(0, c - 1) for c in dc.values())
            scope = scope_of(rows)
            return scope * 100 + 0.05 * s + 100000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts, cur_sum)
        best_cost = cur_cost

        T = 12.0
        cooling = 0.9997
        min_T = 0.02

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.30:
                # Row rebuild in difference-space
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = total_mask ^ old_mask

                new_row = None
                for attempt in range(5):
                    row = [0]
                    x = 0
                    local_mask = 0
                    success = True
                    for j in range(1, row_len):
                        found = False
                        start = x + 1
                        if attempt > 0 and rng.random() < 0.4:
                            start = x + rng.randint(1, 4)
                        for t in range(start, cur_scope + 8):
                            new_mask = 0
                            for a in row:
                                new_mask |= (1 << (t - a))
                            if (new_mask & used_without) or (new_mask & local_mask):
                                continue
                            row.append(t)
                            local_mask |= new_mask
                            x = t
                            found = True
                            break
                        if not found:
                            success = False
                            break
                    if success:
                        new_row = row
                        break

                if new_row is None:
                    continue

                new_diffs = diffs_from_row(new_row)
                new_mask = row_mask(new_row)

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                cur_diffs[i] = new_diffs
                row_masks[i] = new_mask
                total_mask = total_mask ^ old_mask ^ new_mask
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    cur_diffs[i] = old_diffs
                    row_masks[i] = old_mask
                    total_mask = total_mask ^ new_mask ^ old_mask

            elif r < 0.50:
                # Scope compression: reduce max element via Capped Greedy
                max_val = cur_scope
                max_row_idx = -1
                for ri in range(n):
                    if cur_rows[ri][-1] == max_val:
                        max_row_idx = ri
                        break
                if max_row_idx < 0:
                    continue

                i = max_row_idx
                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = total_mask ^ old_mask

                new_row = None
                for shrink in range(1, 8):
                    cap = old_row[-1] - shrink
                    if cap < row_len - 1:
                        break
                    candidate = greedy_row_capped(used_without, cap)
                    if candidate is not None:
                        new_row = candidate
                        break

                if new_row is not None:
                    new_diffs = diffs_from_row(new_row)
                    new_mask = row_mask(new_row)
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    cur_diffs[i] = new_diffs
                    row_masks[i] = new_mask
                    total_mask = total_mask ^ old_mask ^ new_mask
                    new_sum = sum_entries(cur_rows)
                    new_cost = get_cost(cur_rows, diff_counts, new_sum)

                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        cur_scope = scope_of(cur_rows)
                        cur_sum = new_sum
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = cur_scope
                            best_sum = cur_sum
                            best_rows = [list(r) for r in cur_rows]
                            best_diffs = [d[:] for d in cur_diffs]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row
                        cur_diffs[i] = old_diffs
                        row_masks[i] = old_mask
                        total_mask = total_mask ^ new_mask ^ old_mask

            elif r < 0.72:
                # Single difference mutation (difference-space)
                i = rng.randint(0, n - 1)
                pos = rng.randint(0, k - 1)
                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = total_mask ^ old_mask

                if rng.random() < 0.7:
                    delta = rng.randint(-3, 1)
                else:
                    delta = rng.randint(-5, 5)
                new_diff = max(1, old_diffs[pos] + delta)
                if new_diff == old_diffs[pos]:
                    continue

                new_diffs = old_diffs[:]
                new_diffs[pos] = new_diff
                new_row = row_from_diffs(new_diffs)

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                cur_diffs[i] = new_diffs
                row_masks[i] = new_mask
                total_mask = total_mask ^ old_mask ^ new_mask
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    cur_diffs[i] = old_diffs
                    row_masks[i] = old_mask
                    total_mask = total_mask ^ new_mask ^ old_mask

            elif r < 0.85:
                # Swap two differences within a row
                i = rng.randint(0, n - 1)
                p1 = rng.randint(0, k - 1)
                p2 = rng.randint(0, k - 1)
                if p1 == p2:
                    continue

                old_row = list(cur_rows[i])
                old_diffs = cur_diffs[i][:]
                old_mask = row_masks[i]
                used_without = total_mask ^ old_mask

                new_diffs = old_diffs[:]
                new_diffs[p1], new_diffs[p2] = old_diffs[p2], old_diffs[p1]
                if new_diffs == old_diffs:
                    continue

                new_row = row_from_diffs(new_diffs)
                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                cur_diffs[i] = new_diffs
                row_masks[i] = new_mask
                total_mask = total_mask ^ old_mask ^ new_mask
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    cur_diffs[i] = old_diffs
                    row_masks[i] = old_mask
                    total_mask = total_mask ^ new_mask ^ old_mask

            else:
                # Two-row swap: exchange entire rows between positions
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue

                old_row_i = list(cur_rows[i])
                old_row_j = list(cur_rows[j])
                old_diffs_i = cur_diffs[i][:]
                old_diffs_j = cur_diffs[j][:]
                old_mask_i = row_masks[i]
                old_mask_j = row_masks[j]

                used_without_ij = total_mask ^ old_mask_i ^ old_mask_j

                new_row_i = row_from_diffs(old_diffs_j)
                new_row_j = row_from_diffs(old_diffs_i)
                new_mask_i = row_mask(new_row_i)
                new_mask_j = row_mask(new_row_j)

                if (new_mask_i & used_without_ij) or (new_mask_j & used_without_ij) or (new_mask_i & new_mask_j):
                    continue

                remove_row_diffs(diff_counts, old_row_i)
                remove_row_diffs(diff_counts, old_row_j)
                add_row_diffs(diff_counts, new_row_i)
                add_row_diffs(diff_counts, new_row_j)
                cur_rows[i] = new_row_i
                cur_rows[j] = new_row_j
                cur_diffs[i] = old_diffs_j[:]
                cur_diffs[j] = old_diffs_i[:]
                row_masks[i] = new_mask_i
                row_masks[j] = new_mask_j
                total_mask = total_mask ^ old_mask_i ^ old_mask_j ^ new_mask_i ^ new_mask_j
                new_sum = sum_entries(cur_rows)
                new_cost = get_cost(cur_rows, diff_counts, new_sum)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    cur_sum = new_sum
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_sum = cur_sum
                        best_rows = [list(r) for r in cur_rows]
                        best_diffs = [d[:] for d in cur_diffs]
                else:
                    remove_row_diffs(diff_counts, new_row_i)
                    remove_row_diffs(diff_counts, new_row_j)
                    add_row_diffs(diff_counts, old_row_i)
                    add_row_diffs(diff_counts, old_row_j)
                    cur_rows[i] = old_row_i
                    cur_rows[j] = old_row_j
                    cur_diffs[i] = old_diffs_i
                    cur_diffs[j] = old_diffs_j
                    row_masks[i] = old_mask_i
                    row_masks[j] = old_mask_j
                    total_mask = total_mask ^ new_mask_i ^ new_mask_j ^ old_mask_i ^ old_mask_j

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    def final_polish(rows, deadline):
        cur = [list(r) for r in rows]
        n = len(cur)
        cur_masks = [row_mask(r) for r in cur]
        total_mask = 0
        for rm in cur_masks:
            total_mask |= rm
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_scope = scope_of(cur)
            for i in range(n):
                if time.time() >= deadline:
                    break
                cur_scope_i = cur[i][-1]
                if cur_scope_i <= 1:
                    continue
                used_wo = total_mask ^ cur_masks[i]

                new_row = None
                for shrink in range(1, 5):
                    cap = cur_scope_i - shrink
                    if cap < row_len - 1:
                        break
                    candidate = greedy_row_capped(used_wo, cap)
                    if candidate is not None:
                        new_row = candidate
                        break

                if new_row is None:
                    res = find_row_dfs(used_wo, cur_scope_i, cur_scope_i - 1, 30000, min(deadline, time.time() + 0.8))
                    if res is not None:
                        new_row, _ = res

                if new_row is not None:
                    if new_row[-1] < cur[i][-1]:
                        new_mask = row_mask(new_row)
                        cur[i] = new_row
                        cur_masks[i] = new_mask
                        total_mask = total_mask ^ cur_masks[i] ^ new_mask
                        improved = True
                        break
        return cur

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        base_budget = 100000
        for idx in range(n_rows):
            if time.time() > dl:
                break
            remaining_frac = max(0.1, (dl - time.time()) / max(1, (dl - t0)))
            budget = int(base_budget * remaining_frac)
            res = find_row_dfs(used_mask, 130, 130, budget, dl)
            if res is None:
                row = greedy_row(used_mask)
                used_mask |= row_mask(row)
            else:
                row, rm = res
                used_mask |= rm
            rows.append(row)
        while len(rows) < n_rows:
            row = greedy_row(used_mask)
            used_mask |= row_mask(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)

        if time.time() < dl - 3:
            sa_time = dl - time.time() - 1
            sa_iters = max(8000, int(25000 * min(1.0, sa_time / 20)))
            sa_scope, sa_rows = sa_optimize(rows, sa_iters, rng, dl - 1)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        if time.time() < dl - 0.5:
            polished = final_polish(best_rows, dl - 0.5)
            if scope_of(polished) < best_scope:
                best_scope = scope_of(polished)
                best_rows = [r[:] for r in polished]

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 3:
            if first:
                dl = min(DEADLINE - 2, time.time() + 40)
                first = False
            else:
                remaining = DEADLINE - 2 - time.time()
                dl = min(DEADLINE - 2, time.time() + min(20, remaining))
            sc, rows = run(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass

    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask)
        used_mask |= row_mask(row)
        rows.append(row)
    return rows
