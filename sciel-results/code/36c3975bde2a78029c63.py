import time
import random

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 112.0
    row_len = k + 1
    num_diffs_per_row = k * (k + 1) // 2

    def row_diffs_set(row):
        d = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d.add(row[j] - ri)
        return d

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

    def find_row_fast(used_mask, best_max_init, S_cap, node_budget, dl):
        """Find a row with max < best_max_init using bitmask DFS."""
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

    def run(seed, dl):
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        # Build initial rows
        for _ in range(n_rows):
            if time.time() > dl:
                break
            res = find_row_fast(used_mask, 200, 200, 150000, dl)
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

        # Iterative re-optimization passes
        pass_count = 0
        no_improve = 0
        while time.time() < dl and no_improve < 3:
            pass_count += 1
            improved = False
            # Order by descending max for better packing
            order = list(range(n_rows))
            order.sort(key=lambda i: (-rows[i][-1], rng.random()))

            for idx in order:
                if time.time() > dl:
                    break
                r = rows[idx]
                rm = row_mask(r)
                used_without = used_mask & ~rm
                cur = r[-1]
                # Try to find a tighter row
                res = find_row_fast(used_without, cur, cur, 100000, dl)
                if res is not None:
                    nr, nrm = res
                    rows[idx] = nr
                    used_mask = used_without | nrm
                    improved = True
                else:
                    # Try greedy as fallback if DFS fails
                    pass

            if not improved:
                no_improve += 1
                # Perturbation: randomly rebuild one row
                if time.time() > dl:
                    break
                pi = rng.randrange(n_rows)
                rm = row_mask(rows[pi])
                used_without = used_mask & ~rm
                rows[pi] = greedy_row(used_without)
                used_mask = used_without | row_mask(rows[pi])
            else:
                no_improve = 0

            s = scope_of(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 65)
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

    # Last-resort valid greedy
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used_mask)
        used_mask |= row_mask(row)
        rows.append(row)
    return rows
