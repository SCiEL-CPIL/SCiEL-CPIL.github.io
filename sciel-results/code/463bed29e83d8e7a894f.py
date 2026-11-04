import time
import random
import math
import sys


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    L = k + 1  # length of each row
    total_diffs_needed = n_rows * diffs_per_row

    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_list(row):
        """Return list of all positive differences in the row."""
        res = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                res.append(row[j] - row[i])
        return res

    def row_diffs_set(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    # Fast membership using a bytearray sized to a reasonable max scope
    MAX_SCOPE = 2000  # differences won't exceed this in practice for small n_rows,k

    def make_used_array(scope):
        return bytearray(scope + 1)

    def pure_greedy():
        """Pure greedy: pick smallest available value at each step."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def randomized_greedy(window=60):
        """Greedy with shuffled candidate window for diversity."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new = {cand - a for a in row}
                    if not (new & used):
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not (new & used):
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def backtracking_construct(timeout=20.0):
        """
        Backtracking with pruning: build rows one by one, within each row
        try to place entries using MRV-like heuristic (try smaller values first
        but with some randomization). Prune when a placement leads to dead end.
        """
        start = time.time()
        used = set()
        rows = []

        def build_row():
            """Build one row via backtracking."""
            row = [0]
            x = 0

            def backtrack():
                nonlocal x
                if len(row) == L:
                    return True
                # Try candidates in randomized order, biased toward small
                base = x
                cands = list(range(base + 1, base + 40))
                random.shuffle(cands)
                # Also try some larger values
                for _ in range(5):
                    cands.append(base + random.randint(40, 80))
                cands.sort()
                for cand in cands:
                    if cand <= base:
                        continue
                    new = {cand - a for a in row}
                    if new & used:
                        continue
                    # Tentatively place
                    row.append(cand)
                    used |= new
                    old_x = x
                    x = cand
                    if backtrack():
                        return True
                    # Undo
                    x = old_x
                    used -= new
                    row.pop()
                return False

            if backtrack():
                return row
            return None

        for _ in range(n_rows):
            if time.time() - start > timeout:
                return None
            r = build_row()
            if r is None:
                return None
            rows.append(r)
        return rows

    def local_search_rows(rows, time_limit):
        """
        Local search that operates on entire rows.
        Moves:
        1. Perturb a single entry in a row (small shift).
        2. Rebuild a single row from scratch given the used diffs of other rows.
        3. Swap two entries within a row (maintaining order via re-sort won't work
           since order matters for differences... actually differences are symmetric
           in magnitude, so we just need strict increase).
        """
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)

        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        it = 0
        no_improve = 0
        max_no_improve = 50000

        while time.time() < time_limit:
            it += 1

            if no_improve > max_no_improve:
                break

            ri = random.randrange(n_rows)
            move_type = random.random()

            if move_type < 0.5:
                # Single entry perturbation
                ci = random.randrange(1, L)
                old_val = rows[ri][ci]
                old_row = rows[ri]
                old_diffs = row_diff_sets[ri]

                # Generate candidates
                max_shift = max(3, int(0.05 * best_scope))
                candidates = set()
                for d in range(-max_shift, max_shift + 1):
                    nv = old_val + d
                    if nv >= 1:
                        candidates.add(nv)
                for _ in range(3):
                    nv = random.randint(1, best_scope)
                    candidates.add(nv)

                lower = old_row[ci - 1] + 1
                upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
                candidates = [c for c in candidates if lower <= c <= upper]
                if not candidates:
                    no_improve += 1
                    continue

                random.shuffle(candidates)
                accepted = False
                for nv in candidates:
                    if nv == old_val:
                        continue
                    new_row = old_row[:]
                    new_row[ci] = nv
                    new_diffs = row_diffs_set(new_row)

                    if len(new_diffs) != diffs_per_row:
                        continue

                    other_diffs = all_diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue

                    new_scope = max(max(r) for r in rows[:ri] + [new_row] + rows[ri+1:])
                    if new_scope < best_scope:
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row
                        best = [r[:] for r in rows]
                        best_scope = new_scope
                        no_improve = 0
                        accepted = True
                        break

                if not accepted:
                    no_improve += 1

            elif move_type < 0.8:
                # Rebuild row ri from scratch, avoiding all_diffs minus row ri's own diffs
                other_diffs = all_diffs - row_diff_sets[ri]
                # Try to build a new row that avoids other_diffs
                new_row = [0]
                x = 0
                success = True
                while len(new_row) < L:
                    found = False
                    # Try a window of candidates
                    for _attempt in range(200):
                        if _attempt < 50:
                            cand = x + 1 + random.randint(0, 30)
                        else:
                            cand = random.randint(x + 1, best_scope)
                        if cand <= x:
                            continue
                        new = {cand - a for a in new_row}
                        if new & other_diffs:
                            continue
                        if len(new) != len(new_row):  # internal collision
                            continue
                        new_row.append(cand)
                        x = cand
                        found = True
                        break
                    if not found:
                        success = False
                        break

                if success and len(new_row) == L:
                    new_diffs = row_diffs_set(new_row)
                    if len(new_diffs) != diffs_per_row:
                        success = False
                    elif new_diffs & other_diffs:
                        success = False

                if success:
                    new_scope = max(max(r) for r in rows[:ri] + [new_row] + rows[ri+1:])
                    if new_scope < best_scope:
                        all_diffs = (all_diffs - row_diff_sets[ri]) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row
                        best = [r[:] for r in rows]
                        best_scope = new_scope
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            else:
                # Pairwise row swap attempt: try to find two rows that can be
                # rebuilt to reduce scope. For now, just perturb both.
                ri2 = random.randrange(n_rows)
                if ri2 == ri:
                    no_improve += 1
                    continue
                ci = random.randrange(1, L)
                old_val = rows[ri][ci]
                old_row = rows[ri]
                old_diffs = row_diff_sets[ri]

                max_shift = max(3, int(0.05 * best_scope))
                candidates = set()
                for d in range(-max_shift, max_shift + 1):
                    nv = old_val + d
                    if nv >= 1:
                        candidates.add(nv)
                for _ in range(3):
                    nv = random.randint(1, best_scope)
                    candidates.add(nv)

                lower = old_row[ci - 1] + 1
                upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
                candidates = [c for c in candidates if lower <= c <= upper]
                if not candidates:
                    no_improve += 1
                    continue

                random.shuffle(candidates)
                accepted = False
                for nv in candidates:
                    if nv == old_val:
                        continue
                    new_row = old_row[:]
                    new_row[ci] = nv
                    new_diffs = row_diffs_set(new_row)

                    if len(new_diffs) != diffs_per_row:
                        continue

                    other_diffs = all_diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue

                    new_scope = max(max(r) for r in rows[:ri] + [new_row] + rows[ri+1:])
                    if new_scope < best_scope:
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row
                        best = [r[:] for r in rows]
                        best_scope = new_scope
                        no_improve = 0
                        accepted = True
                        break

                if not accepted:
                    no_improve += 1

            if best_scope <= 112:
                break

        return best, best_scope

    def simulated_annealing(rows, time_limit):
        """Simulated annealing with scope as energy."""
        rows = [r[:] for r in rows]
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        current_scope = best_scope

        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        T0 = 5.0
        T_min = 0.01
        total_iters = 0
        stuck = 0
        max_stuck = 30000

        while time.time() < time_limit:
            frac = min(1.0, total_iters / 100000.0)
            T = T0 * (T_min / T0) ** frac
            if T < T_min:
                T = T_min
            total_iters += 1

            ri = random.randrange(n_rows)
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_diffs = row_diff_sets[ri]

            max_shift = max(5, int(0.08 * current_scope))
            shift = random.randint(-max_shift, max_shift // 2)
            nv = old_val + shift

            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else current_scope
            if nv < lower:
                nv = lower
            if nv > upper:
                nv = upper
            if nv == old_val:
                continue

            new_row = old_row[:]
            new_row[ci] = nv
            new_diffs = row_diffs_set(new_row)

            if len(new_diffs) != diffs_per_row:
                continue

            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                continue

            new_max = max(new_row)
            new_scope = max(current_scope, new_max)

            if new_scope <= current_scope:
                accept = True
            else:
                delta = new_scope - current_scope
                if T > 0:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False

            if accept:
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[ri] = new_diffs
                rows[ri] = new_row
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    stuck = 0
                else:
                    stuck += 1

                if stuck > max_stuck:
                    stuck = 0
            else:
                stuck += 1

            if best_scope <= 112:
                break

        return best_rows, best_scope

    # ---------- Main search ----------
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)

    # Phase 1: Try backtracking construction (may find better starting point)
    bt_rows = backtracking_construct(timeout=15.0)
    if bt_rows is not None:
        bt_scope = scope_of(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope

    # Phase 2: Randomized greedy restarts
    candidates_pool = [(best_scope, best_rows)]
    attempts = 0
    while time.time() < deadline - 60.0 and attempts < 40:
        attempts += 1
        rows = randomized_greedy(window=30 + random.randint(0, 60))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            candidates_pool.insert(0, (sc, rows))
            if best_scope <= 112:
                break
        elif len(candidates_pool) < 6 and sc < best_scope + 25:
            candidates_pool.append((sc, rows))

    candidates_pool.sort(key=lambda x: x[0])

    # Phase 3: Simulated annealing on top candidates
    for sc, start_rows in candidates_pool[:3]:
        if time.time() >= deadline - 30.0:
            break
        improved, new_sc = simulated_annealing(start_rows, time_limit=deadline - 30.0)
        if new_sc < best_scope:
            best_rows = improved
            best_scope = new_sc
            if best_scope <= 112:
                break

    # Phase 4: Local search polish
    if time.time() < deadline - 10.0:
        improved, sc = local_search_rows(best_rows, time_limit=deadline - 10.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc

    # Final validation
    all_diffs = set()
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    # Should not happen; but if it does, fall back to greedy
                    best_rows = pure_greedy()
                    return best_rows
                all_diffs.add(d)

    return best_rows
