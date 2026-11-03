import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin before 120s limit
    row_len = k + 1

    # --- Bitmask helpers for fast conflict detection ---
    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

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

    def get_cost(rows, dc):
        collisions = sum(max(0, c - 1) for c in dc.values())
        scope = scope_of(rows)
        s = sum_entries(rows)
        # Primary: scope, Secondary: sum of entries (packing density), Penalty: collisions
        return scope + 0.01 * s + 1000 * collisions

    # --- Greedy initial construction ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    # --- Targeted DFS to find compact row given used diffs ---
    def find_row_dfs(used_diffs, max_val, rng, node_budget=2000, time_limit=0.05):
        start = time.time()
        best_row = None
        best_max = max_val + 1
        nodes = [0]

        def dfs(row, local_used, last):
            nonlocal best_row, best_max
            if time.time() - start > time_limit or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                if m < best_max:
                    best_max = m
                    best_row = row[:]
                return
            lo = last + 1
            hi = min(best_max - 1, max_val)
            if lo > hi:
                return
            # Try smaller values first for compactness
            candidates = list(range(lo, hi + 1))
            if len(candidates) > 50 and rng.random() < 0.5:
                # Randomize order for diversity
                rng.shuffle(candidates)
                # But keep first few in order for greedy bias
                candidates = candidates[:3] + candidates[3:]
            for x in candidates:
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    local_used.update(new_diffs)
                    dfs(row, local_used, x)
                    row.pop()
                    local_used.difference_update(new_diffs)
                    if time.time() - start > time_limit or nodes[0] >= node_budget:
                        return

        dfs([0], set(), 0)
        return best_row

    # --- SA optimization with bitmask-accelerated moves ---
    def sa_optimize(rows, deadline, polish=False):
        rng = random.Random()
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope

        diff_counts = build_diff_counts(cur_rows)
        row_masks = [row_mask(r) for r in cur_rows]
        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        if polish:
            T = 0.3
            cooling = 0.99995
            min_T = 0.005
        else:
            T = 5.0
            cooling = 0.9997
            min_T = 0.02

        n = len(cur_rows)

        while time.time() < deadline:
            r = rng.random()

            if r < 0.35:
                # Row rebuild via DFS
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_mask = row_masks[i]

                remove_row_diffs(diff_counts, old_row)
                used_diffs = set(diff_counts.keys())

                # Try to find compact row
                new_row = find_row_dfs(used_diffs, cur_scope, rng, node_budget=1500, time_limit=0.03)
                if new_row is None:
                    new_row = find_row_dfs(used_diffs, cur_scope + 8, rng, node_budget=2000, time_limit=0.04)

                if new_row is None:
                    add_row_diffs(diff_counts, old_row)
                    continue

                new_mask = row_mask(new_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                row_masks[i] = new_mask

                new_cost = get_cost(cur_rows, diff_counts)
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row
                    row_masks[i] = old_mask

            elif r < 0.55:
                # Scope compression: target the max element
                max_val = cur_scope
                max_row_idx = -1
                max_pos_idx = -1
                for ri in range(n):
                    for pi in range(k, 0, -1):
                        if cur_rows[ri][pi] == max_val:
                            max_row_idx = ri
                            max_pos_idx = pi
                            break
                    if max_row_idx >= 0:
                        break

                if max_row_idx < 0:
                    continue

                i = max_row_idx
                j = max_pos_idx
                old_val = cur_rows[i][j]
                prev_val = cur_rows[i][j - 1]

                if old_val <= prev_val + 1:
                    continue

                new_val = rng.randint(prev_val + 1, old_val - 1)

                remove_row_diffs(diff_counts, cur_rows[i])
                cur_rows[i][j] = new_val
                add_row_diffs(diff_counts, cur_rows[i])
                row_masks[i] = row_mask(cur_rows[i])

                new_cost = get_cost(cur_rows, diff_counts)
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, cur_rows[i])
                    cur_rows[i][j] = old_val
                    add_row_diffs(diff_counts, cur_rows[i])
                    row_masks[i] = row_mask(cur_rows[i])

            elif r < 0.75:
                # Single element move
                r_idx = rng.randint(0, n - 1)
                i_idx = rng.randint(1, row_len - 1)

                lower = cur_rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = cur_rows[r_idx][i_idx + 1] - 1
                else:
                    upper = cur_scope + 5

                if lower > upper:
                    continue

                new_val = rng.randint(lower, upper)
                old_val = cur_rows[r_idx][i_idx]

                if new_val == old_val:
                    continue

                remove_row_diffs(diff_counts, cur_rows[r_idx])
                cur_rows[r_idx][i_idx] = new_val
                add_row_diffs(diff_counts, cur_rows[r_idx])
                row_masks[r_idx] = row_mask(cur_rows[r_idx])

                new_cost = get_cost(cur_rows, diff_counts)
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, cur_rows[r_idx])
                    cur_rows[r_idx][i_idx] = old_val
                    add_row_diffs(diff_counts, cur_rows[r_idx])
                    row_masks[r_idx] = row_mask(cur_rows[r_idx])

            else:
                # Pair swap: swap two elements in the same row (if order preserved)
                r_idx = rng.randint(0, n - 1)
                i_idx = rng.randint(1, row_len - 2)
                j_idx = rng.randint(i_idx + 1, row_len - 1)

                # Swap only if it maintains increasing order
                if cur_rows[r_idx][i_idx] + cur_rows[r_idx][j_idx] > cur_rows[r_idx][j_idx - 1] + cur_rows[r_idx][i_idx + 1] if j_idx + 1 < row_len else False:
                    continue
                # Simpler: just try random perturbation of two positions
                old_a = cur_rows[r_idx][i_idx]
                old_b = cur_rows[r_idx][j_idx]

                # Try to find new values that maintain order
                lower_a = cur_rows[r_idx][i_idx - 1] + 1
                upper_a = cur_rows[r_idx][j_idx] - 1
                lower_b = cur_rows[r_idx][i_idx] + 1
                upper_b = cur_rows[r_idx][j_idx + 1] - 1 if j_idx + 1 < row_len else cur_scope + 5

                if lower_a > upper_a or lower_b > upper_b:
                    continue

                new_a = rng.randint(lower_a, upper_a)
                new_b = rng.randint(max(lower_b, new_a + 1), upper_b)

                if new_a == old_a and new_b == old_b:
                    continue

                remove_row_diffs(diff_counts, cur_rows[r_idx])
                cur_rows[r_idx][i_idx] = new_a
                cur_rows[r_idx][j_idx] = new_b
                add_row_diffs(diff_counts, cur_rows[r_idx])
                row_masks[r_idx] = row_mask(cur_rows[r_idx])

                new_cost = get_cost(cur_rows, diff_counts)
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    cur_scope = scope_of(cur_rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = cur_scope
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, cur_rows[r_idx])
                    cur_rows[r_idx][i_idx] = old_a
                    cur_rows[r_idx][j_idx] = old_b
                    add_row_diffs(diff_counts, cur_rows[r_idx])
                    row_masks[r_idx] = row_mask(cur_rows[r_idx])

            T *= cooling
            if T < min_T:
                T = min_T

        return best_rows, best_scope

    # Main loop
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)

    # Run multiple SA rounds with different initializations
    while time.time() < DEADLINE:
        remaining = DEADLINE - time.time()
        if remaining < 2.0:
            break

        # Choose initialization strategy
        roll = random.random()
        if roll < 0.4:
            # Use best rows as starting point
            init_rows = [r[:] for r in best_rows]
        elif roll < 0.7:
            # Random initialization within bound
            bound = best_max + 10
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        else:
            # Greedy with random perturbation
            init_rows = greedy_construct()
            for i in range(len(init_rows)):
                for j in range(1, len(init_rows[i])):
                    if random.random() < 0.2:
                        init_rows[i][j] += random.randint(-3, 3)
                        init_rows[i][j] = max(init_rows[i][j], init_rows[i][j-1] + 1)

        # Determine SA time budget
        sa_time = min(remaining - 1.0, 8.0)
        if sa_time < 1.0:
            break

        sa_deadline = time.time() + sa_time
        local_rows, local_max = sa_optimize(init_rows, sa_deadline, polish=False)

        if local_max < best_max:
            best_rows = local_rows
            best_max = local_max

    # Final polish phase if time remains
    remaining = DEADLINE - time.time()
    if remaining > 3.0:
        polish_deadline = time.time() + min(remaining - 0.5, 10.0)
        polished_rows, polished_max = sa_optimize(best_rows, polish_deadline, polish=True)
        if polished_max < best_max:
            best_rows = polished_rows
            best_max = polished_max

    return best_rows
