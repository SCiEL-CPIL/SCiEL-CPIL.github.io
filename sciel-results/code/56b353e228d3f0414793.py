import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 118.0
    row_len = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    S_cap = 130  # Hard upper bound on entry values to keep bitmasks manageable

    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def greedy_row(used_mask, rng=None, randomize=False):
        row = [0]
        x = 0
        local_mask = 0
        while len(row) < row_len:
            found = False
            if randomize and rng and len(row) > 1:
                skip = rng.randint(0, 2)
                start = x + 1 + skip
            else:
                start = x + 1
            for t in range(start, S_cap + 1):
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

    def get_cost(rows, dc):
        scope = scope_of(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 1000 * collisions

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

    def try_swap_two_entries(cur_rows, diff_counts, rng):
        """
        Structural move: pick two entries (possibly in different rows) and swap them,
        if the resulting rows remain strictly increasing and do not create collisions.
        Returns (applied, old_row1, old_row2, new_row1, new_row2, i1, i2) for proper revert.
        """
        i1 = rng.randint(0, n_rows - 1)
        j1 = rng.randint(1, row_len - 1)
        i2 = rng.randint(0, n_rows - 1)
        j2 = rng.randint(1, row_len - 1)
        
        if i1 == i2 and j1 == j2:
            return None
        
        row1 = cur_rows[i1]
        row2 = cur_rows[i2]
        v1 = row1[j1]
        v2 = row2[j2]
        
        if v1 == v2:
            return None
        
        new_row1 = list(row1)
        new_row2 = list(row2)
        new_row1[j1] = v2
        new_row2[j2] = v1
        
        for idx in range(len(new_row1) - 1):
            if new_row1[idx] >= new_row1[idx + 1]:
                return None
        for idx in range(len(new_row2) - 1):
            if new_row2[idx] >= new_row2[idx + 1]:
                return None
        
        mask1_new = row_mask(new_row1)
        mask2_new = row_mask(new_row2)
        
        used_without = 0
        for ri in range(n_rows):
            if ri != i1 and ri != i2:
                used_without |= row_mask(cur_rows[ri])
        
        if (mask1_new & used_without) or (mask2_new & used_without):
            return None
        if (mask1_new & mask2_new):
            return None
        
        remove_row_diffs(diff_counts, row1)
        remove_row_diffs(diff_counts, row2)
        add_row_diffs(diff_counts, new_row1)
        add_row_diffs(diff_counts, new_row2)
        
        cur_rows[i1] = new_row1
        cur_rows[i2] = new_row2
        
        return (True, list(row1), list(row2), new_row1, new_row2, i1, i2)

    def compress_rows_polish(rows, diff_counts, dl):
        """
        Multi-pass compression: repeatedly try to reduce the last entry of each row
        by finding a smaller valid value that doesn't create collisions.
        """
        improved = True
        passes = 0
        while improved and time.time() < dl:
            improved = False
            passes += 1
            for i in range(n_rows):
                if time.time() > dl:
                    break
                old_row = list(rows[i])
                old_last = old_row[-1]
                
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(rows[ri])
                
                # Try to find a row with smaller last entry via DFS
                dfs_dl = min(dl, time.time() + 0.08)
                res = find_row_dfs(used_without, old_last, old_last - 1, 3000, dfs_dl)
                if res is not None:
                    new_row, _ = res
                    if new_row[-1] < old_last:
                        remove_row_diffs(diff_counts, old_row)
                        add_row_diffs(diff_counts, new_row)
                        rows[i] = new_row
                        improved = True
        return passes

    # Phase 1: Greedy construction
    best_rows = None
    best_scope = float('inf')
    rng = random.Random(42)
    
    n_seeds = 60
    for seed in range(n_seeds):
        if time.time() >= DEADLINE - 25:
            break
        r = random.Random(seed)
        used_mask = 0
        rows = []
        success = True
        for _ in range(n_rows):
            res = greedy_row(used_mask, r, randomize=(seed > 5))
            if res is None:
                success = False
                break
            row, mask = res
            rows.append(row)
            used_mask |= mask
        if success:
            scope = scope_of(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Fallback simple greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                v = row[-1] + 1
                while True:
                    valid = True
                    for prev in row:
                        d = v - prev
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for prev in row[:-1]:
                            used_diffs.add(v - prev)
                        break
                    v += 1
            rows.append(row)
        return rows

    # Phase 2: Simulated Annealing with structural swap move
    if time.time() < DEADLINE - 15:
        cur_rows = [list(r) for r in best_rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope
        
        diff_counts = build_diff_counts(cur_rows)
        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 8.0
        cooling = 0.999
        min_T = 0.02
        iters = 0
        max_iters = 80000
        sa_deadline = DEADLINE - 15
        
        while time.time() < sa_deadline and T > min_T and iters < max_iters:
            iters += 1
            r = rng.random()
            
            if r < 0.20:
                # Row rebuild
                i = rng.randint(0, n_rows - 1)
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                res = row_rebuild_greedy(used_without, min(S_cap, cur_scope + 5), rng)
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

            elif r < 0.35:
                # Structural swap move with proper revert
                swap_result = try_swap_two_entries(cur_rows, diff_counts, rng)
                if swap_result is not None:
                    applied, old_r1, old_r2, new_r1, new_r2, si1, si2 = swap_result
                    new_cost = get_cost(cur_rows, diff_counts)
                    if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                        cur_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = scope_of(cur_rows)
                            best_rows = [list(r) for r in cur_rows]
                    else:
                        # Revert swap
                        remove_row_diffs(diff_counts, new_r1)
                        remove_row_diffs(diff_counts, new_r2)
                        add_row_diffs(diff_counts, old_r1)
                        add_row_diffs(diff_counts, old_r2)
                        cur_rows[si1] = old_r1
                        cur_rows[si2] = old_r2

            elif r < 0.55:
                # Scope compression via DFS on the row with largest last entry
                max_val = max(r[-1] for r in cur_rows)
                max_row_idx = next(ri for ri in range(n_rows) if cur_rows[ri][-1] == max_val)
                
                i = max_row_idx
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                dfs_dl = min(sa_deadline - 1, time.time() + 0.3)
                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 8000, dfs_dl)
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

            elif r < 0.75:
                # Single element move
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                old_val = old_row[j]
                for delta in range(-4, 5):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if j > 0 and new_val <= old_row[j - 1]:
                        continue
                    if j < len(old_row) - 1 and new_val >= old_row[j + 1]:
                        continue
                    
                    new_row = list(old_row)
                    new_row[j] = new_val
                    new_mask = row_mask(new_row)
                    
                    if (new_mask & used_without) == 0:
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
                            break
                        else:
                            remove_row_diffs(diff_counts, new_row)
                            add_row_diffs(diff_counts, old_row)
                            cur_rows[i] = old_row
                            break

            else:
                # Multi-row shuffle: rebuild two random rows simultaneously
                idxs = rng.sample(range(n_rows), 2)
                i, j = idxs
                old_row_i = list(cur_rows[i])
                old_row_j = list(cur_rows[j])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i and ri != j:
                        used_without |= row_mask(cur_rows[ri])
                
                res_i = row_rebuild_greedy(used_without, min(S_cap, cur_scope + 3), rng)
                if res_i is None:
                    continue
                new_row_i, mask_i = res_i
                # Update used mask for second row
                used_without2 = used_without | mask_i
                res_j = row_rebuild_greedy(used_without2, min(S_cap, cur_scope + 3), rng)
                if res_j is None:
                    continue
                new_row_j, mask_j = res_j
                
                if (mask_i & mask_j) != 0:
                    continue
                
                remove_row_diffs(diff_counts, old_row_i)
                remove_row_diffs(diff_counts, old_row_j)
                add_row_diffs(diff_counts, new_row_i)
                add_row_diffs(diff_counts, new_row_j)
                cur_rows[i] = new_row_i
                cur_rows[j] = new_row_j
                new_cost = get_cost(cur_rows, diff_counts)

                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = scope_of(cur_rows)
                        best_rows = [list(r) for r in cur_rows]
                else:
                    remove_row_diffs(diff_counts, new_row_i)
                    remove_row_diffs(diff_counts, new_row_j)
                    add_row_diffs(diff_counts, old_row_i)
                    add_row_diffs(diff_counts, old_row_j)
                    cur_rows[i] = old_row_i
                    cur_rows[j] = old_row_j

            T *= cooling

    # Phase 3: Final local polish - aggressive multi-pass compression
    if time.time() < DEADLINE - 2:
        polish_rows = [list(r) for r in best_rows]
        polish_dc = build_diff_counts(polish_rows)
        polish_dl = DEADLINE - 2
        
        # Multiple passes of compression
        for pass_num in range(3):
            if time.time() > polish_dl:
                break
            compress_rows_polish(polish_rows, polish_dc, polish_dl)
        
        # Final single-entry micro-optimization pass
        if time.time() < polish_dl:
            for i in range(n_rows):
                if time.time() > polish_dl:
                    break
                old_row = list(polish_rows[i])
                used_without = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_without |= row_mask(polish_rows[ri])
                
                # Try each position, each small delta
                for j in range(1, row_len):
                    if time.time() > polish_dl:
                        break
                    old_val = old_row[j]
                    for delta in range(-5, 6):
                        if delta == 0:
                            continue
                        new_val = old_val + delta
                        if new_val <= 0:
                            continue
                        if j > 0 and new_val <= old_row[j - 1]:
                            continue
                        if j < row_len - 1 and new_val >= old_row[j + 1]:
                            continue
                        
                        new_row = list(old_row)
                        new_row[j] = new_val
                        new_mask = row_mask(new_row)
                        
                        if (new_mask & used_without) == 0:
                            # Check if this reduces the row's max
                            if new_row[-1] < old_row[-1]:
                                remove_row_diffs(polish_dc, old_row)
                                add_row_diffs(polish_dc, new_row)
                                polish_rows[i] = new_row
                                old_row = new_row
                                break

        if scope_of(polish_rows) < best_scope:
            best_rows = polish_rows
            best_scope = scope_of(best_rows)

    return best_rows
