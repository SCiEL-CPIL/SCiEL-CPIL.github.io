import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 132.0  # safety margin before 150s kill

    d = k * (k + 1) // 2  # differences per row
    max_diff = 200  # upper bound on any difference we care about

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def greedy_init():
        """Greedy construction with smarter skip: try multiple candidates per step."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                for trial in range(x + 1, x + 50):
                    new = {trial - a for a in row}
                    if not new & used:
                        row.append(trial)
                        used |= new
                        x = trial
                        found = True
                        break
                if not found:
                    # Fallback: just keep incrementing
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(12345)

    def rebuild_row(used_other, jitter=0.0, max_val=None, local_rng=None):
        """Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed."""
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_diffs = set()
        x = 0
        search_end = max_val if max_val is not None else (best_scope + 30)
        while len(row) < k + 1:
            found = False
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=8):
        """Reduce scope by locally re-optimizing each row, bottleneck-prioritized."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 12 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Bottleneck-prioritized order: largest rows first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                # More attempts with varied jitter for better exploration
                for attempt in range(40):
                    if attempt < 15:
                        jitter = 0.0
                    elif attempt < 25:
                        jitter = 0.05 + 0.02 * (attempt - 15)
                    else:
                        jitter = 0.25 + 0.03 * (attempt - 25)
                    new_row = rebuild_row(used_other, jitter=jitter, max_val=best_max - 1)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        """Perturb by rebuilding random rows with high jitter."""
        rows = [list(r) for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(20):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=best_scope + 30)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_scope + 40)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    def targeted_descent(rows, budget=3.0):
        """Try to reduce scope by shrinking the largest entry in max rows."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                # Try interior positions
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(15):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
                continue

            # Try all values from lower to old_val-1 (descending for early exit)
            for new_val in range(old_val - 1, lower - 1, -1):
                row[pos] = new_val
                scope = full_validate(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    else:
                        # Scope didn't improve but might be same; try next lower
                        continue
                row[pos] = old_val
                break

            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    def exact_local_polish(rows, budget=4.0):
        """Exact local solve: for each position in each row, try all valid lower values."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        best_cur = [r[:] for r in cur]
        best_sc = cur_scope
        improved = True

        while improved and time.time() - start < budget:
            improved = False
            for ri in range(n_rows):
                if time.time() - start > budget:
                    break
                row = cur[ri]
                for pos in range(len(row) - 1, 0, -1):  # from end to start
                    if time.time() - start > budget:
                        break
                    old_val = row[pos]
                    lo = row[pos - 1] + 1
                    hi = row[pos + 1] - 1 if pos < len(row) - 1 else cur_scope + 10
                    if hi <= lo:
                        continue
                    # Try all values from old_val-1 down to lo
                    for new_val in range(old_val - 1, lo - 1, -1):
                        row[pos] = new_val
                        scope = full_validate(cur)
                        if scope is not None:
                            if scope < best_sc:
                                best_sc = scope
                                best_cur = [r[:] for r in cur]
                            improved = True
                            break
                        else:
                            row[pos] = old_val
                            break
                    if improved:
                        break
                if improved:
                    cur = [r[:] for r in best_cur]
                    cur_scope = best_sc

        return best_cur, best_sc

    def molecule_anneal(rows, cur_scope, budget=5.0):
        """Metropolis with proper revert on invalid moves."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_sc = cur_scope
        best_local = [r[:] for r in rows]
        best_local_sc = cur_scope
        T = 0.6

        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            row_snapshot = cur[ri][:]
            row = cur[ri]

            if rng.random() < 0.5 and len(row) > 2:
                pos = rng.randint(1, len(row) - 2)
                old_val = row[pos]
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                row[pos] = new_val
            else:
                delta = rng.randint(-3, 3)
                if delta == 0:
                    continue
                shifted = [v + delta for v in row]
                if min(shifted) < 0:
                    continue
                anchor = min(shifted)
                new_row = [v - anchor for v in shifted]
                if new_row[0] != 0 or any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                row[:] = new_row

            new_scope = full_validate(cur)
            if new_scope is not None:
                if new_scope < best_local_sc:
                    best_local_sc = new_scope
                    best_local = [r[:] for r in cur]
                delta_sc = new_scope - cur_sc
                if delta_sc <= 0 or rng.random() < (2.718281828 ** (-delta_sc / max(T, 0.01))):
                    cur_sc = new_scope
                else:
                    cur[ri] = row_snapshot
            else:
                cur[ri] = row_snapshot

            T *= 0.9998

        return best_local, best_local_sc

    # Phase 1: ILS with row reconstruction + local polish
    max_il_iter = 300
    for it in range(max_il_iter):
        if time.time() > deadline - 22:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            if rng.random() < 0.7:
                cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))
            else:
                i = rng.randrange(n_rows)
                j = rng.randrange(n_rows)
                if i != j:
                    cur[i], cur[j] = cur[j], cur[i]

        cur_scope = full_validate(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Local polish (bottleneck-prioritized)
        if time.time() < deadline - 16:
            polished = local_polish(cur)
            ps = full_validate(polished)
            if ps is not None and ps < best_scope:
                best_scope = ps
                best_rows = [r[:] for r in polished]
                cur = [r[:] for r in polished]

        # Targeted descent
        if time.time() < deadline - 12:
            desc_rows, desc_scope = targeted_descent(cur, budget=1.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                cur = [r[:] for r in desc_rows]

    # Phase 2: Fine-grained annealing
    if time.time() < deadline - 10:
        start = time.time()
        cur = [r[:] for r in best_rows]
        cur_sc = best_scope
        best_local = [r[:] for r in best_rows]
        best_local_sc = best_scope
        T = 0.8
        while time.time() - start < 6.0:
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            cur[ri][pos] = new_val
            new_scope = full_validate(cur)
            if new_scope is not None:
                if new_scope < best_local_sc:
                    best_local_sc = new_scope
                    best_local = [r[:] for r in cur]
                delta = new_scope - cur_sc
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_sc = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val
            T *= 0.9998
        if best_local_sc < best_scope:
            best_scope = best_local_sc
            best_rows = best_local

    # Phase 3: Exact local polish (try all lower values for each position)
    if time.time() < deadline - 6:
        polished, ps = exact_local_polish(best_rows, budget=3.0)
        if ps < best_scope:
            best_scope = ps
            best_rows = polished

    # Phase 4: Final local polish pass
    if time.time() < deadline - 4:
        polished = local_polish(best_rows, max_passes=6)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    # Phase 5: Final targeted descent
    if time.time() < deadline - 2:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=1.5)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Phase 6: Final exact local polish with remaining time
    if time.time() < deadline - 0.5:
        remaining = deadline - time.time() - 0.5
        if remaining > 1.0:
            polished, ps = exact_local_polish(best_rows, budget=remaining)
            if ps < best_scope:
                best_scope = ps
                best_rows = polished

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
