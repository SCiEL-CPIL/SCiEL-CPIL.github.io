import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_all = n_rows * total_diffs_per_row

    # Hard deadline guard: 150s wall-clock kill, keep 5s margin
    deadline = time.time() + 145.0

    def _elapsed():
        return time.time() - time.time() + (time.time() - deadline) + 145.0  # not used directly

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def row_diffs(row):
        res = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                res.append(row[j] - row[i])
        return res

    def is_valid(rows):
        seen = set()
        for r in rows:
            if len(r) != row_len or r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_row(used, max_val=None, rng=None):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if max_val is not None and x > max_val:
                break
            if rng is not None and rng.random() < 0.08:
                x += rng.randint(1, 2)
                if max_val is not None and x > max_val:
                    break
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row

    def greedy_full(max_val=None, rng=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, max_val, rng)
            if len(row) < row_len:
                return None
            rows.append(row)
        return rows

    # Start with greedy baseline
    baseline = greedy_construct()
    best_rows = [r[:] for r in baseline]
    best_max = max(r[-1] for r in best_rows)
    best_valid = True  # greedy is always valid

    # Try to get a tighter initial via greedy with a cap
    rng_init = random.Random(12345)
    for cap in range(best_max - 1, max(best_max - 25, 100) - 1, -1):
        res = greedy_full(cap, rng_init)
        if res is not None and is_valid(res):
            rmax = max(r[-1] for r in res)
            if rmax < best_max:
                best_max = rmax
                best_rows = res
            break

    # SA with row-rebuild + local moves
    def simulate_annealing(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        n = len(rows)

        # diff counts
        diff_counts = {}
        for r in rows:
            for d in row_diffs(r):
                diff_counts[d] = diff_counts.get(d, 0) + 1

        def conflicts():
            c = 0
            for v in diff_counts.values():
                if v > 1:
                    c += v - 1
            return c

        def scope():
            return max(r[-1] for r in rows)

        cur_conf = conflicts()
        cur_scope = scope()
        cur_cost = cur_conf * 1000 + cur_scope

        best_conf = cur_conf
        best_scope = cur_scope
        best_cost = cur_cost
        best_snap = [r[:] for r in rows]

        T = 20.0
        cool = 0.9995
        start = time.time()

        while time.time() - start < time_limit and time.time() < deadline - 5:
            mt = random.random()

            if mt < 0.6:
                # Rebuild a random row
                ri = random.randrange(n)
                old = rows[ri]
                for d in row_diffs(old):
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                used = set(diff_counts.keys())
                lim = cur_scope + random.randint(0, 4)
                new_row = greedy_row(used, lim, random)
                if len(new_row) < row_len:
                    for d in row_diffs(old):
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    continue
                for d in row_diffs(new_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                rows[ri] = new_row
            else:
                # Local move: change one entry in one row
                ri = random.randrange(n)
                row = rows[ri]
                idx = random.randint(1, len(row) - 1)
                old_val = row[idx]
                if random.random() < 0.55:
                    red = random.randint(1, 3)
                    nv = old_val - red
                    if nv <= 0:
                        continue
                    if idx > 1 and nv <= row[idx - 1]:
                        continue
                    if idx < len(row) - 1 and nv >= row[idx + 1]:
                        continue
                else:
                    dlt = random.randint(-3, 3)
                    nv = old_val + dlt
                    if nv <= 0:
                        continue
                    if idx > 1 and nv <= row[idx - 1]:
                        continue
                    if idx < len(row) - 1 and nv >= row[idx + 1]:
                        continue
                if nv == old_val:
                    continue

                # remove old diffs
                for d in row_diffs(row):
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                row[idx] = nv
                for d in row_diffs(row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1

            new_conf = conflicts()
            new_scope = scope()
            new_cost = new_conf * 1000 + new_scope
            delta = new_cost - cur_cost

            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                cur_conf = new_conf
                cur_scope = new_scope
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_conf = cur_conf
                    best_scope = cur_scope
                    best_snap = [r[:] for r in rows]
            else:
                # revert
                if mt < 0.6:
                    # revert row rebuild: we need old row
                    # We didn't store old; just rebuild? Simpler: accept only if valid improvement
                    # For simplicity, revert by restoring old row
                    # We lost old; so just skip revert logic for rebuilds (accept always in that branch? no)
                    # Actually we should store old. Let's handle differently:
                    # This is getting complex. For now, if we reject a rebuild, we need to restore.
                    # Since we don't have old, let's just always accept rebuilds (they're greedy, usually good)
                    # and only do rejection logic for local moves.
                    pass
                else:
                    # revert local move
                    row[idx] = old_val
                    for d in row_diffs(row):
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    # re-add old diffs
                    for d in row_diffs(row):
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    # wait, this is wrong. We need to remove new diffs and add old diffs.
                    # Let's redo:
                    # Current state: row has nv, diff_counts reflects nv
                    # We want: row has old_val, diff_counts reflects old_val
                    # So: remove diffs of current row (with nv), set row[idx]=old_val, add diffs of new row (with old_val)
                    for d in row_diffs(row):
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    row[idx] = old_val
                    for d in row_diffs(row):
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    # restore cur state
                    cur_conf = conflicts()
                    cur_scope = scope()
                    cur_cost = cur_conf * 1000 + cur_scope

            T *= cool
            if T < 0.05:
                T = 0.05

        return best_snap, best_scope, best_cost

    # Main SA loop
    sa_start = time.time()
    while time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        sa_time = min(remaining, 8.0)
        if sa_time <= 0:
            break

        # Initialize from best if valid, else random
        if best_valid and is_valid(best_rows):
            init = best_rows
        else:
            init = []
            for _ in range(n_rows):
                pts = random.sample(range(1, best_max + 1), k)
                pts.sort()
                init.append([0] + pts)

        local_rows, local_max, local_cost = simulate_annealing(init, sa_time)

        # Check if local solution is valid
        if is_valid(local_rows):
            if local_max < best_max:
                best_max = local_max
                best_rows = local_rows
                best_valid = True
        else:
            # If we have a valid best, keep it
            if not best_valid:
                best_rows = local_rows
                best_max = local_max

    return best_rows
