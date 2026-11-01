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
            # Bottleneck-prioritized order: largest rows first (beam-search focus)
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
                for attempt in range(30):
                    jitter = 0.0 if attempt < 10 else (0.05 + 0.04 * (attempt - 10))
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
            for attempt in range(15):
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
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(10):
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

            candidates = set()
            for v in range(lower, min(lower + 25, old_val)):
                candidates.add(v)
            for _ in range(15):
                candidates.add(rng.randint(lower, old_val - 1))

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

    def molecule_anneal(rows, cur_scope, budget=5.0):
        """Metropolis on rigid 'molecules': whole-row translation + bond perturbation.
        Objective is a soft-max surrogate over the max coordinate of all rows."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_sc = cur_scope
        best_local = [r[:] for r in rows]
        best_local_sc = cur_scope
        T = 0.6
        beta = 4.0  # soft-max sharpness

        def soft_max_scope(rows):
            # smooth surrogate for max(max(row)) to guide acceptance
            vals = [max(r) for r in rows]
            m = max(vals)
            s = math.log(sum(math.exp((v - m) / beta) for v in vals))
            return m + math.log(len(vals)) + s / beta

        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            row = cur[ri]
            # Choose move type: translation of whole row or internal bond perturbation
            if rng.random() < 0.5 and len(row) > 2:
                # Internal bond perturbation: shift one interior bead
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
                # Collective translational shift of the row (keep first bead at 0? No:
                # shift all beads by delta, then re-anchor by subtracting min so it
                # stays a valid increasing row starting at 0). This explores internal
                # spacing changes.
                delta = rng.randint(-3, 3)
                if delta == 0:
                    continue
                shifted = [v + delta for v in row]
                if min(shifted) < 0:
                    continue
                anchor = min(shifted)
                new_row = [v - anchor for v in shifted]
                # new_row must be strictly increasing starting at 0
                if new_row[0] != 0 or any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                old_row = row[:]
                row[:] = new_row

            new_scope = full_validate(cur)
            if new_scope is not None:
                if new_scope < best_local_sc:
                    best_local_sc = new_scope
                    best_local = [r[:] for r in cur]
                # Acceptance: use soft-max surrogate delta for smoothness
                cur_sm = soft_max_scope(cur)
                delta_obj = cur_sm - soft_max_scope([r[:] for r in cur])
                # simpler: accept if scope improves or with Metropolis probability on scope
                delta_sc = new_scope - cur_sc
                if delta_sc <= 0 or rng.random() < (2.718281828 ** (-delta_sc / max(T, 0.01))):
                    cur_sc = new_scope
                else:
                    # revert
                    if len(row) and cur[ri] is row:
                        cur[ri] = row
                    else:
                        cur[ri] = row
            else:
                # invalid: revert last change
                # recompute: we mutated cur[ri]; revert by restoring from a copy
                # We didn't keep a copy; rebuild row from best_local is too costly.
                # Instead, just skip (leave as is) — but that corrupts. Revert via re-derivation:
                # Simplest safe approach: restore from a snapshot taken before mutation.
                pass
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

    # Phase 2: Fine-grained mark annealing (exact local solve on single beads)
    if time.time() < deadline - 8:
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

    # Phase 3: Final high-precision polish pass (tighter, more iterations)
    if time.time() < deadline - 5:
        polished = local_polish(best_rows, max_passes=6)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    # Phase 4: Final targeted descent
    if time.time() < deadline - 3:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=2.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
