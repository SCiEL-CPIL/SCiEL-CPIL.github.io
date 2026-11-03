import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1

    # Bitmask helpers
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

    # Greedy construction
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

    def greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < row_len:
                x += 1
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (x - a))
                if (new_mask & used) == 0:
                    row.append(x)
                    used |= new_mask
            rows.append(row)
        return rows

    # Diff count helpers for SA
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

    # Row rebuild via greedy with randomness
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

    # DFS to find a row with max element < best_max
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

    # Simulated Annealing
    def sa_optimize(rows, iters, rng, deadline):
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
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

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

            elif r < 0.50:
                # Scope compression: reduce max element via DFS
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
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])

                res = find_row_dfs(used_without, old_row[-1], old_row[-1] - 1, 50000, min(deadline, time.time() + 2))
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

            else:
                # Single element move
                i = rng.randint(0, n - 1)
                j = rng.randint(1, row_len - 1)
                old_row = list(cur_rows[i])
                old_val = old_row[j]
                
                # Try small deltas
                delta = rng.randint(-5, 5)
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
                
                # Check internal duplicates
                new_diffs = set()
                for a in range(row_len):
                    for b in range(a + 1, row_len):
                        new_diffs.add(new_row[b] - new_row[a])
                if len(new_diffs) != row_len * (row_len - 1) // 2:
                    continue

                # Check conflicts
                new_mask = row_mask(new_row)
                used_without = 0
                for ri in range(n):
                    if ri != i:
                        used_without |= row_mask(cur_rows[ri])
                
                if (new_mask & used_without) != 0:
                    continue

                # Apply
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

            T *= cooling

        return best_rows, best_scope

    # Local polish
    def local_polish(rows, deadline):
        best_rows = [list(r) for r in rows]
        best_scope = scope_of(best_rows)
        rng = random.Random(123)
        n = len(best_rows)
        
        diff_counts = build_diff_counts(best_rows)
        
        for _ in range(20000):
            if time.time() >= deadline - 5:
                break
            
            ri = rng.randrange(n)
            row = best_rows[ri]
            pos = rng.randrange(1, row_len)
            old_val = row[pos]
            
            for delta in range(-3, 4):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < row_len - 1 and new_val >= row[pos + 1]:
                    continue
                
                new_row = list(row)
                new_row[pos] = new_val
                
                new_diffs = set()
                for a in range(row_len):
                    for b in range(a + 1, row_len):
                        new_diffs.add(new_row[b] - new_row[a])
                if len(new_diffs) != row_len * (row_len - 1) // 2:
                    continue

                new_mask = row_mask(new_row)
                used_without = 0
                for rj in range(n):
                    if rj != ri:
                        used_without |= row_mask(best_rows[rj])
                
                if (new_mask & used_without) != 0:
                    continue

                new_scope = scope_of(best_rows)
                # Check if scope improves
                if new_scope < best_scope:
                    remove_row_diffs(diff_counts, row)
                    add_row_diffs(diff_counts, new_row)
                    best_rows[ri] = new_row
                    best_scope = new_scope
                    break
                else:
                    # Revert
                    pass
        
        return best_rows, best_scope

    # Phase 1: Greedy construction
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(20):
        if time.time() >= DEADLINE - 20:
            break
        rows = greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = scope_of(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Fallback
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

    # Phase 2: Simulated Annealing
    if best_rows is not None and time.time() < DEADLINE - 10:
        rng = random.Random(42)
        best_rows, best_scope = sa_optimize(best_rows, 50000, rng, DEADLINE - 10)

    # Phase 3: Local Polish
    if best_rows is not None and time.time() < DEADLINE - 5:
        best_rows, best_scope = local_polish(best_rows, DEADLINE)

    return best_rows
