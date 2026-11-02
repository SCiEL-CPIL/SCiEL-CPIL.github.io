import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0

    d = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
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
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def rebuild_row(used_other, max_val=None, jitter=0.0, rng_local=None):
        if rng_local is None:
            rng_local = random.Random()
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            search_end = max_val if max_val else (best_scope + 40)
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and rng_local.random() < jitter:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def exact_local_polish(rows, budget=5.0):
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return None, None
        improved = True
        while improved and time.time() - start < budget:
            improved = False
            for i in range(n_rows):
                if time.time() - start > budget:
                    break
                row = cur[i]
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(cur[j])
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start > budget:
                        break
                    old_val = row[pos]
                    lo = row[pos - 1] + 1
                    hi = row[pos + 1] - 1 if pos < len(row) - 1 else cur_scope
                    for new_val in range(old_val - 1, lo - 1, -1):
                        if new_val < lo:
                            break
                        row[pos] = new_val
                        scope = full_validate(cur)
                        if scope is not None and scope < cur_scope:
                            cur_scope = scope
                            improved = True
                            break
                        row[pos] = old_val
        if cur_scope < get_scope(rows):
            return cur, cur_scope
        return None, None

    def kick(rows, num_kicks=1, rng_local=None):
        if rng_local is None:
            rng_local = random.Random()
        rows = [list(r) for r in rows]
        indices = rng_local.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(20):
                jitter = 0.2 + 0.05 * attempt
                new_row = rebuild_row(used_other, max_val=best_scope + 30, jitter=jitter, rng_local=rng_local)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, max_val=best_scope + 40, jitter=0.0, rng_local=rng_local)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    def targeted_descent(rows, budget=3.0, rng_local=None):
        if rng_local is None:
            rng_local = random.Random()
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return None, None
        improved = False

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng_local.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                pos2 = rng_local.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(10):
                    nv = rng_local.randint(lo2, hi2)
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

            candidates = set()
            for v in range(lower, min(lower + 25, old_val)):
                candidates.add(v)
            for _ in range(15):
                candidates.add(rng_local.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val

            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    def gap_swap_move(rows, cur_scope, rng_local=None):
        if rng_local is None:
            rng_local = random.Random()
        gaps = []
        for r in rows:
            g = [r[i+1] - r[i] for i in range(len(r)-1)]
            gaps.append(g)

        i, j = rng_local.sample(range(n_rows), 2)

        pos = rng_local.randint(0, len(gaps[i]) - 1)

        old_gap_i = gaps[i][pos]
        old_gap_j = gaps[j][pos]
        if old_gap_i == old_gap_j:
            return None

        gaps[i][pos] = old_gap_j
        gaps[j][pos] = old_gap_i

        new_rows = [list(r) for r in rows]
        for idx in [i, j]:
            new_rows[idx] = [0]
            for g in gaps[idx]:
                new_rows[idx].append(new_rows[idx][-1] + g)

        new_scope = full_validate(new_rows)
        if new_scope is not None and new_scope < cur_scope:
            return new_rows, new_scope

        gaps[i][pos] = old_gap_i
        gaps[j][pos] = old_gap_j
        return None

    def gap_local_search(rows, budget=2.0, rng_local=None):
        if rng_local is None:
            rng_local = random.Random()
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return None, None

        improved = True
        while improved and time.time() - start < budget:
            improved = False
            for _ in range(150):
                if time.time() - start > budget:
                    break
                result = gap_swap_move(cur, cur_scope, rng_local)
                if result is not None:
                    cur, cur_scope = result
                    improved = True
                    break

        if cur_scope < get_scope(rows):
            return cur, cur_scope
        return None, None

    # Phase 1: ILS with row reconstruction
    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    max_il_iter = 300
    for it in range(max_il_iter):
        if time.time() > deadline - 25:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            cur = kick(cur, num_kicks=rng.choice([1, 1, 1, 2]), rng_local=rng)

        cur_scope = full_validate(cur)
        if cur_scope is None:
            continue

        # Exact local polish
        if time.time() < deadline - 20:
            polished, pol_scope = exact_local_polish(cur, budget=3.0)
            if polished is not None:
                cur = polished
                cur_scope = pol_scope

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Gap-based local search
        if time.time() < deadline - 18:
            gap_rows, gap_scope = gap_local_search(cur, budget=1.5, rng_local=rng)
            if gap_rows is not None and gap_scope < best_scope:
                best_scope = gap_scope
                best_rows = gap_rows

        # Targeted descent
        if time.time() < deadline - 15:
            desc_rows, desc_scope = targeted_descent(cur, budget=2.0, rng_local=rng)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    # Phase 2: Fine-grained SA on single elements
    if time.time() < deadline - 20:
        cur = [r[:] for r in best_rows]
        cur_scope = full_validate(cur)
        if cur_scope is None:
            cur = [r[:] for r in best_rows]
            cur_scope = best_scope

        T = 0.9
        sa_steps = 80000
        for step in range(sa_steps):
            if time.time() > deadline - 8:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 20
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope = full_validate(cur)

            if new_scope is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val

            T *= 0.99997

    # Final exact polish with remaining time
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            polished, pol_scope = exact_local_polish(best_rows, budget=min(remaining, 8.0))
            if polished is not None:
                best_rows = polished
                best_scope = pol_scope

    # Final targeted descent
    if time.time() < deadline - 3:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=2.0, rng_local=rng)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Final gap-based polish
    if time.time() < deadline - 2:
        gap_rows, gap_scope = gap_local_search(best_rows, budget=1.0, rng_local=rng)
        if gap_rows is not None and gap_scope < best_scope:
            best_scope = gap_scope
            best_rows = gap_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
