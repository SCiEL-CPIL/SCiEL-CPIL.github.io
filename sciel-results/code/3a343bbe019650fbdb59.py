"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    L = k + 1  # length of each row

    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = set()
        for row in rows:
            diffs = row_diffs_set(row)
            if len(diffs) != diffs_per_row:
                return False
            if diffs & used:
                return False
            used |= diffs
        return True

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def randomized_greedy(window=80):
        """Greedy with shuffled candidate window."""
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
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def backtrack_construct(used, max_scope, depth=0):
        """
        Backtracking search for a single row given already-used differences.
        Returns a list of L increasing integers starting at 0, or None if no solution.
        """
        if depth >= L:
            return [0] * L if L == 1 else None  # placeholder, we build incrementally

        row = [0]
        x = 0

        def rec(row, x):
            if len(row) == L:
                return row[:]
            for cand in range(x + 1, max_scope + 1):
                new = {cand - a for a in row}
                if new & used:
                    continue
                row.append(cand)
                result = rec(row, cand)
                if result is not None:
                    return result
                row.pop()
            return None

        return rec(row, x)

    def simulated_annealing(rows, max_iters, time_limit):
        """
        Simulated annealing on the set of rows.
        Moves: change one entry in one row.
        Accept if valid and scope decreases, or with probability exp(-delta/T).
        """
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])

        # Precompute per-row diff sets
        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        T = 50.0
        T_min = 0.5
        cooling = 0.999

        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            T = max(T_min, T * cooling)

            ri = random.randrange(n)
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_diffs = row_diff_sets[ri]

            # Generate candidate values
            candidates = set()
            for d in range(-15, 16):
                nv = old_val + d
                if nv >= 1:
                    candidates.add(nv)
            for _ in range(10):
                nv = random.randint(1, best_scope + 20)
                candidates.add(nv)

            # Enforce ordering
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope + 20
            candidates = [c for c in candidates if lower <= c <= upper]
            if not candidates:
                continue

            random.shuffle(candidates)
            for nv in candidates:
                if nv == old_val:
                    continue
                new_row = old_row[:]
                new_row[ci] = nv
                new_diffs = row_diffs_set(new_row)

                # Check internal consistency
                if len(new_diffs) != diffs_per_row:
                    continue

                # Check no collision with other rows
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue

                # Valid move
                new_scope = max(best_scope, max(new_row))
                delta = new_scope - best_scope

                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    # Accept move
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    row_diff_sets[ri] = new_diffs
                    rows[ri] = new_row

                    if new_scope < best_scope:
                        best = [r[:] for r in rows]
                        best_scope = new_scope
                    break

        return best, best_scope

    def reduce_scope(rows, time_limit):
        """Try to reduce the scope by moving the largest entries down."""
        rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])

        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        # Sort rows by their max value descending, try to reduce the largest first
        for _ in range(100):
            if time.time() > time_limit:
                break
            # Find the row with the current maximum
            max_row_idx = max(range(n), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]

            # Try to reduce the last element
            old_last = row[-1]
            for nv in range(old_last - 1, row[-2], -1):
                new_row = row[:-1] + [nv]
                new_diffs = row_diffs_set(new_row)
                if len(new_diffs) != diffs_per_row:
                    continue
                old_diffs = row_diff_sets[max_row_idx]
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue
                # Valid
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[max_row_idx] = new_diffs
                rows[max_row_idx] = new_row
                best_scope = scope_of(rows)
                break

        return rows, best_scope

    # ---------- Main search ----------
    # Start with pure greedy
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)

    # Phase 1: Randomized greedy restarts
    attempts = 0
    while time.time() < deadline - 50.0 and attempts < 200:
        attempts += 1
        rows = randomized_greedy(window=40 + random.randint(0, 80))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 2: Simulated annealing on best and a few alternatives
    pool = [best_rows]
    for _ in range(8):
        if time.time() >= deadline - 30.0:
            break
        rows = randomized_greedy(window=40 + random.randint(0, 80))
        sc = scope_of(rows)
        if sc <= best_scope + 30:
            pool.append(rows)

    for start in pool:
        if time.time() >= deadline - 15.0:
            break
        improved, sc = simulated_annealing(start, max_iters=10000, time_limit=deadline - 15.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 3: Scope reduction pass
    if time.time() < deadline - 5.0:
        reduced, sc = reduce_scope(best_rows, deadline - 5.0)
        if sc < best_scope:
            best_rows = reduced
            best_scope = sc

    return best_rows
