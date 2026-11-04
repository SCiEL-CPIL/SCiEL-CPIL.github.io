"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2

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
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def randomized_greedy(window=80):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
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

    def backtracking_construct(max_scope, time_limit):
        """Try to construct a valid set with all entries <= max_scope using backtracking."""
        if time.time() > time_limit:
            return None

        used = set()
        rows = []

        def backtrack(row_idx, current_row, pos_in_row):
            if time.time() > time_limit:
                return False

            if pos_in_row == k + 1:
                rows.append(current_row[:])
                if row_idx == n_rows - 1:
                    return True
                return backtrack(row_idx + 1, [0], 1)

            if row_idx == n_rows:
                return True

            # Determine the lower bound for the next value
            if pos_in_row == 1:
                lower = 1
            else:
                lower = current_row[-1] + 1

            # Upper bound
            upper = max_scope

            # Try values in a randomized order for diversity
            candidates = list(range(lower, upper + 1))
            if len(candidates) > 30:
                # Sample a subset to keep backtracking tractable
                random.shuffle(candidates)
                candidates = candidates[:30]
                # Always include the smallest few
                candidates = sorted(set(candidates[:10] + candidates[-10:]))
            else:
                random.shuffle(candidates)

            for val in candidates:
                new_diffs = {val - a for a in current_row}
                if new_diffs & used:
                    continue
                # Check internal consistency (all diffs within row must be distinct)
                all_row_diffs = set()
                for i in range(len(current_row) + 1):
                    for j in range(i + 1, len(current_row) + 1):
                        if j == len(current_row):
                            d = val - current_row[i]
                        else:
                            d = current_row[j] - current_row[i]
                        if d in all_row_diffs:
                            break
                        all_row_diffs.add(d)
                    else:
                        continue
                    break
                else:
                    # All diffs are distinct within the row
                    current_row.append(val)
                    used |= new_diffs
                    if backtrack(row_idx, current_row, pos_in_row + 1):
                        return True
                    current_row.pop()
                    used -= new_diffs

            return False

        if backtrack(0, [0], 1):
            return rows
        return None

    def simulated_annealing(rows, max_iters, time_limit):
        """Simulated annealing on the full configuration."""
        rows = [r[:] for r in rows]
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        current_scope = best_scope

        n = len(rows)
        L = len(rows[0])

        # Precompute
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

            # Choose a random move
            move_type = random.random()
            ri = random.randrange(n)
            ci = random.randrange(1, L)

            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_diffs = row_diff_sets[ri]
            other_diffs = all_diffs - old_diffs

            # Generate new value
            if move_type < 0.5:
                # Small perturbation
                nv = old_val + random.randint(-5, 5)
            elif move_type < 0.8:
                # Medium perturbation
                nv = random.randint(1, current_scope)
            else:
                # Large perturbation - try to reduce scope
                nv = random.randint(1, max(1, current_scope - 10))

            # Enforce ordering
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

            # Check internal consistency
            if len(new_diffs) != diffs_per_row:
                continue

            # Check no collision with other rows
            if new_diffs & other_diffs:
                continue

            # Compute new scope
            new_scope = max(max(r) for r in rows[:ri] + [new_row] + rows[ri+1:])

            # Acceptance criterion
            delta = new_scope - current_scope
            if delta <= 0 or random.random() < (1.0 / (1.0 + delta / T)):
                # Accept the move
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diff_sets[ri] = new_diffs
                rows[ri] = new_row
                current_scope = new_scope

                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]

        return best_rows, best_scope

    def multi_pass_reduction(rows, time_limit):
        """Multi-pass: repeatedly try to reduce the largest entries."""
        rows = [r[:] for r in rows]
        best_scope = scope_of(rows)

        for _ in range(200):
            if time.time() > time_limit:
                break

            # Find row with max value
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_idx]
            old_last = row[-1]

            row_diff_sets = [row_diffs_set(r) for r in rows]
            all_diffs = set()
            for s in row_diff_sets:
                all_diffs |= s
            old_diffs = row_diff_sets[max_idx]
            other_diffs = all_diffs - old_diffs

            # Try to reduce last element
            found = False
            for nv in range(old_last - 1, row[-2], -1):
                new_row = row[:-1] + [nv]
                new_diffs = row_diffs_set(new_row)
                if len(new_diffs) != diffs_per_row:
                    continue
                if new_diffs & other_diffs:
                    continue
                # Valid reduction
                all_diffs = (all_diffs - old_diffs) | new_diffs
                rows[max_idx] = new_row
                best_scope = scope_of(rows)
                found = True
                break

            if not found:
                # Try reducing other elements in the max row
                for ci in range(L - 2, 0, -1):
                    if time.time() > time_limit:
                        break
                    old_val = row[ci]
                    lower = row[ci - 1] + 1
                    upper = row[ci + 1] - 1
                    for nv in range(old_val - 1, lower - 1, -1):
                        new_row = row[:]
                        new_row[ci] = nv
                        new_diffs = row_diffs_set(new_row)
                        if len(new_diffs) != diffs_per_row:
                            continue
                        if new_diffs & other_diffs:
                            continue
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        rows[max_idx] = new_row
                        best_scope = scope_of(rows)
                        found = True
                        break
                    if found:
                        break

            if not found:
                break

        return rows, best_scope

    # ---------- Main search ----------
    # Phase 0: Get a baseline
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)

    # Phase 1: Randomized greedy restarts
    attempts = 0
    while time.time() < deadline - 50.0 and attempts < 80:
        attempts += 1
        rows = randomized_greedy(window=40 + random.randint(0, 80))
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            if best_scope <= 112:
                break

    # Phase 2: Try backtracking with tighter scope
    if time.time() < deadline - 40.0:
        # Try to construct with scope = best_scope - 1
        target = best_scope - 1
        result = backtracking_construct(target, deadline - 40.0)
        if result is not None:
            sc = scope_of(result)
            if sc < best_scope:
                best_rows = result
                best_scope = sc

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15.0:
        improved, sc = simulated_annealing(best_rows, max_iters=20000, time_limit=deadline - 15.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc

    # Phase 4: Multi-pass reduction
    if time.time() < deadline - 5.0:
        reduced, sc = multi_pass_reduction(best_rows, deadline - 5.0)
        if sc < best_scope:
            best_rows = reduced
            best_scope = sc

    return best_rows
