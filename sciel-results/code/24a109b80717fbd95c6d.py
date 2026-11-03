import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1
    num_diffs = row_len * (row_len - 1) // 2  # 15 for k=5

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def row_diffs_list(row):
        """Return sorted list of all positive differences in a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.append(row[j] - ri)
        return sorted(diffs)

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

    def greedy_row_from_diffs(used_mask, target_max, rng, max_attempts=50):
        """Build a row by choosing differences directly, then reconstructing coordinates."""
        for _ in range(max_attempts):
            # Choose 15 distinct positive differences not in used_mask
            chosen = []
            # Start with small differences
            candidates = list(range(1, target_max + 1))
            random.Random(rng).shuffle(candidates)
            for c in candidates:
                if (1 << c) & used_mask:
                    continue
                chosen.append(c)
                if len(chosen) == num_diffs:
                    break
            if len(chosen) < num_diffs:
                continue
            # Now we need to find coordinates [0, a1, a2, ..., a5] such that
            # all 15 pairwise differences are exactly the chosen set.
            # This is a constraint satisfaction problem.
            # Try DFS on coordinates
            found = _find_coords_from_diffs(chosen, row_len)
            if found is not None:
                row = found
                rm = row_mask(row)
                if (rm & used_mask) == 0:
                    return row, rm
        return None

    def _find_coords_from_diffs(diffs, n):
        """Given a set of n*(n-1)/2 differences, find coordinates [0, x1, ..., x_{n-1}]."""
        diff_set = set(diffs)
        if len(diff_set) != len(diffs):
            return None
        # The largest difference must be x_{n-1} - 0
        max_d = max(diff_set)
        # Try to build the row greedily
        row = [0]
        remaining = set(diff_set)
        # x1 must be in diff_set
        # Use backtracking
        def backtrack(row, remaining):
            if len(row) == n:
                return row[:]
            last = row[-1]
            # Next element x must satisfy: x - a_i for all a_i in row are in remaining
            # and x - 0 = x is in remaining
            # x > last
            # x must be such that all (x - a_i) are in remaining
            # Try candidates: x = last + d for some d, or x such that x is in remaining
            # Better: x must be in remaining (since x - 0 = x)
            # Also x - row[-1] must be in remaining
            # Let's try values of x from last+1 to max_d
            for x in range(last + 1, max_d + 1):
                if x not in remaining:
                    continue
                new_remaining = set(remaining)
                valid = True
                for a in row:
                    d = x - a
                    if d not in new_remaining:
                        valid = False
                        break
                    new_remaining.discard(d)
                if not valid:
                    continue
                row.append(x)
                result = backtrack(row, new_remaining)
                if result is not None:
                    return result
                row.pop()
            return None
        return backtrack([0], diff_set)

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

    def find_row_diffset_dfs(used_mask, best_max_init, S_cap, node_budget, dl):
        """Search in difference-space: choose 15 differences, then solve for coordinates."""
        best = [None]
        best_max = [best_max_init]
        nodes = [0]
        target_diffs = num_diffs  # 15

        def dfs_diffs(chosen, remaining_pool, last_chosen, current_max):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(chosen) == target_diffs:
                # Try to find coordinates
                coords = _find_coords_from_diffs(chosen, row_len)
                if coords is not None:
                    rm = row_mask(coords)
                    if (rm & used_mask) == 0:
                        m = coords[-1]
                        if m < best_max[0]:
                            best_max[0] = m
                            best[0] = coords[:]
                return
            # Next difference must be > last_chosen and not in used_mask
            lo = last_chosen + 1
            hi = min(best_max[0], S_cap)
            if lo > hi:
                return
            for x in range(lo, hi + 1):
                if (1 << x) & used_mask:
                    continue
                if x in chosen:
                    continue
                chosen.append(x)
                dfs_diffs(chosen, remaining_pool, x, max(current_max, x))
                chosen.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return

        dfs_diffs([], set(), 0, 0)
        if best[0] is None:
            return None
        return best[0], row_mask(best[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

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
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope

        diff_counts = build_diff_counts(cur_rows)

        def get_cost(rows, dc):
            scope = scope_of(rows)
            collisions = sum(max(0, c - 1) for c in dc.values())
            return scope + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 5.0
        cooling = 0.9995
        min_T = 0.02

        for it in range(iters):
            if time.time() > deadline:
                break

            r = rng.random()

            if r < 0.20:
                # Row rebuild using difference-space search
                i = rng.randint(0, n - 1)
                old_row = list(cur_rows[i])
                old_mask = row_mask(old_row)
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                # Try diffset-based DFS first
                res = find_row_diffset_dfs(used_without, old_row[-1] + 2, old_row[-1] + 2, 30000, min(deadline, time.time() + 1.5))
                if res is None:
                    # Fall back to coordinate DFS
                    res = find_row_dfs(used_without, old_row[-1] + 2, old_row[-1] + 2, 30000, min(deadline, time.time() + 1.5))
                if res is None:
                    res = row_rebuild_greedy(used_without, cur_scope + 3, rng)
                if res is None:
                    continue
                new_row, new_mask = res

                remove_row_diffs(diff_counts, old_row)
                add_row_diffs(diff_counts, new_row)
                cur_rows[i] = new_row
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row

            elif r < 0.40:
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
                old_mask = row_mask(old_row)
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                # Try diffset-based DFS for tighter row
                res = find_row_diffset_dfs(used_without, old_row[-1], old_row[-1] - 1, 40000, min(deadline, time.time() + 2))
                if res is None:
                    res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 40000, min(deadline, time.time() + 2))
                if res is not None:
                    new_row, new_mask = res
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    new_cost = get_cost(cur_rows, diff_counts)

                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = scope_of(cur_rows)
                            best_rows = [list(r) for r in cur_rows]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row

            elif r < 0.60:
                # Swap two elements within a row (preserving order)
                i = rng.randint(0, n - 1)
                j1 = rng.randint(1, row_len - 2)
                j2 = rng.randint(j1 + 1, row_len - 1)
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                # Try swapping values at positions j1 and j2, but maintain order
                # Actually, let's try a local move: shift one element
                j = rng.randint(1, row_len - 1)
                delta = rng.choice([-1, -2, -3, 1, 2, 3])
                new_val = old_row[j] + delta
                lo = old_row[j - 1] + 1
                hi = old_row[j + 1] - 1 if j < row_len - 1 else cur_scope + 2
                if new_val < lo or new_val > hi:
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
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row

            else:
                # Single element move (larger range)
                i = rng.randint(0, n - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                lo = old_row[j - 1] + 1
                hi = old_row[j + 1] - 1 if j < row_len - 1 else cur_scope + 5
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
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row)
                    add_row_diffs(diff_counts, old_row)
                    cur_rows[i] = old_row

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    def polish(rows, deadline):
        """High-precision final polish: try to reduce scope by 1 via targeted row replacement."""
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope
        diff_counts = build_diff_counts(cur_rows)
        rng = random.Random(12345)

        def get_cost(rows, dc):
            scope = scope_of(rows)
            collisions = sum(max(0, c - 1) for c in dc.values())
            return scope + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        # Try to find a row with max entry == cur_scope and replace it with one having max entry < cur_scope
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Find rows with maximum entry
            max_rows = [i for i in range(len(cur_rows)) if cur_rows[i][-1] == cur_scope]
            if not max_rows:
                break
            i = rng.choice(max_rows)
            old_row = list(cur_rows[i])
            used_without = 0
            for ri in range(len(cur_rows)):
                if ri != i:
                    used_without |= row_mask(cur_rows[ri])

            # Try to find a row with max entry <= cur_scope - 1
            res = find_row_diffset_dfs(used_without, cur_scope, cur_scope - 1, 20000, min(deadline, time.time() + 0.5))
            if res is None:
                res = find_row_dfs(used_without, cur_scope, cur_scope - 1, 20000, min(deadline, time.time() + 0.5))
            if res is not None:
                new_row, new_mask = res
                if new_row[-1] < cur_scope:
                    remove_row_diffs(diff_counts, old_row)
                    add_row_diffs(diff_counts, new_row)
                    cur_rows[i] = new_row
                    new_cost = get_cost(cur_rows, diff_counts)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                    else:
                        remove_row_diffs(diff_counts, new_row)
                        add_row_diffs(diff_counts, old_row)
                        cur_rows[i] = old_row

        return best_scope, best_rows

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        for _ in range(n_rows):
            if time.time() > dl:
                break
            # Try diffset-based search first for tighter rows
            res = find_row_diffset_dfs(used_mask, 150, 150, 50000, dl)
            if res is None:
                res = find_row_dfs(used_mask, 150, 150, 50000, dl)
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
        if time.time() < dl - 8:
            sa_scope, sa_rows = sa_optimize(rows, k, 8000, rng, dl - 6)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

        # Final polish pass
        if time.time() < dl - 2:
            pol_scope, pol_rows = polish(best_rows, dl - 1)
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
                dl = min(DEADLINE - 3, time.time() + 45)
                first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 15)
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
