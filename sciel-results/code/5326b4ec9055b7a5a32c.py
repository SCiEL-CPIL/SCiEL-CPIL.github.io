import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1

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

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    def build_diff_counts(rows):
        counts = Counter()
        for r in rows:
            for i in range(len(r)):
                ri = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ri
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def remove_row_diffs(dc, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                dc[d] -= 1
                if dc[d] == 0:
                    del dc[d]

    def add_row_diffs(dc, row):
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                dc[d] = dc.get(d, 0) + 1

    def row_rebuild_greedy(used_mask, max_scope, rng, attempts=3):
        for attempt in range(attempts):
            row = [0]
            x = 0
            local_mask = 0
            for j in range(1, row_len):
                found = False
                start = x + 1
                if attempt > 0 and rng.random() < 0.3:
                    start = x + rng.randint(1, 3)
                for t in range(start, max_scope + 1):
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
            return row, row_mask(row)
        return None

    def sa_optimize(rows, k, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        cur_sum = sum_entries(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope
        best_sum = cur_sum

        diff_counts = build_diff_counts(cur_rows)
        row_masks = [row_mask(r) for r in cur_rows]

        def compute_used_without(i):
            u = 0
            for ri in range(n):
                if ri != i:
                    u |= row_masks[ri]
            return u

        def get_cost(rows, dc, s):
            scope = scope_of(rows)
            collisions = sum(max(0, c - 1) for c in dc.values())
            return scope + 0.01 * s + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts, cur_sum)
        best_cost = cur_cost

        T = 5.0
        cooling = 0.9997
        min_T = 0.05

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.25:
                # Row rebuild
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                res = row_rebuild_greedy(used_without, cur_scope + 3, rng)
                if res is None:
                    continue
                new_row, new_mask = res

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
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
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    row_masks[i] = old_mask

            elif r < 0.50:
                # Scope compression: reduce max element
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
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                # Try to find a tighter row via DFS
                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 50000, min(deadline, time.time() + 2))
                if res is not None:
                    new_row, new_mask = res
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    row_masks[i] = new_mask
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
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row
                        row_masks[i] = old_mask

            else:
                # Single element move
                i = rng.randint(0, n - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]
                used_without = compute_used_without(i)

                lo = old_row[j - 1] + 1
                hi = old_row[j + 1] - 1 if j < row_len - 1 else cur_scope + 2
                if lo >= hi:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_row[j]:
                    continue

                new_row = list(old_row)
                new_row[j] = new_val
                if new_row[j] <= new_row[j-1] or (j < row_len-1 and new_row[j] >= new_row[j+1]):
                    continue

                new_mask = row_mask(new_row)
                if new_mask & used_without:
                    continue

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask
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
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    row_masks[i] = old_mask

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    def polish_pass(rows, deadline):
        """Final high-precision polish: try to reduce scope by 1 via targeted DFS."""
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        n = len(cur_rows)
        rng = random.Random(42)
        
        # Try reducing scope by 1
        for target_scope in range(cur_scope - 1, max(cur_scope - 10, 100), -1):
            if time.time() > deadline:
                break
            
            # Find rows that exceed target_scope
            for i in range(n):
                if cur_rows[i][-1] > target_scope:
                    # Try to rebuild this row to fit within target_scope
                    used_without = 0
                    for ri in range(n):
                        if ri != i:
                            used_without |= row_mask(cur_rows[ri])
                    
                    # Use DFS with tighter bound
                    res = find_row_dfs(used_without, target_scope, target_scope, 100000, min(deadline, time.time() + 3))
                    if res is not None:
                        new_row, new_mask = res
                        cur_rows[i] = new_row
                        new_scope = scope_of(cur_rows)
                        if new_scope <= target_scope:
                            return new_scope, cur_rows
        
        return cur_scope, cur_rows

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        for _ in range(n_rows):
            if time.time() > dl:
                break
            res = find_row_dfs(used_mask, 200, 200, 100000, dl)
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

        # SA refinement
        if time.time() < dl - 5:
            sa_scope, sa_rows = sa_optimize(rows, k, 5000, rng, dl - 2)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]
        
        # Final polish pass
        if time.time() < dl - 2:
            pol_scope, pol_rows = polish_pass(best_rows, dl - 1)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = [r[:] for r in pol_rows]

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 50)
                first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 12)
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
