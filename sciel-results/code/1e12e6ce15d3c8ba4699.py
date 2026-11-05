import time
import random
import heapq
import bisect
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 142.0

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used

    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True

    def _scope(rows):
        if not rows:
            return 0
        return max(r[-1] for r in rows)

    def _col_wise_construct(seed, target_scope=None, use_randomization=True):
        """
        Column-wise earliest-feasible construction.
        For each column j (1..k), assign values to all n_rows rows simultaneously,
        choosing the smallest feasible value for each row at that column.
        This naturally enforces cross-row difference uniqueness.
        """
        rng = random.Random(42 + seed)
        rows = [[0] for _ in range(n_rows)]
        used = set()

        for col in range(1, k + 1):
            if time.time() > deadline:
                return None

            # For this column, find feasible values for each row
            # Sort rows by their current last value to process in order
            row_order = list(range(n_rows))
            if use_randomization and rng.random() < 0.3:
                rng.shuffle(row_order)
            else:
                row_order.sort(key=lambda i: rows[i][-1])

            assigned = {}
            for ri in row_order:
                row = rows[ri]
                last_val = row[-1]
                start = last_val + 1
                if target_scope is not None:
                    end = min(target_scope + 1, start + 200)
                else:
                    end = start + 200

                best_x = None
                candidates = []
                for x in range(start, end):
                    # Check strict increase within row
                    if x <= last_val:
                        continue
                    # Check differences with previous elements in this row
                    new_diffs = set()
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            break
                        new_diffs.add(d)
                    else:
                        # All differences are new
                        candidates.append(x)
                        if best_x is None:
                            best_x = x
                        if len(candidates) >= 3 and use_randomization:
                            break

                if best_x is None:
                    # Fallback: scan further
                    for x in range(start, start + 500):
                        new_diffs = set()
                        valid = True
                        for i in range(len(row)):
                            d = x - row[i]
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            best_x = x
                            break

                if best_x is None:
                    return None

                # Add randomness: pick from top candidates
                if use_randomization and len(candidates) > 1 and rng.random() < 0.2:
                    best_x = candidates[rng.randint(0, min(2, len(candidates) - 1))]

                rows[ri].append(best_x)
                # Update used differences
                for i in range(len(row)):
                    used.add(best_x - row[i])

        return rows

    def _element_local_search(rows, deadline, rng, max_iter=30000):
        """
        Local search that tries to reduce individual elements while maintaining validity.
        Focuses on reducing the maximum entry first, then other large entries.
        """
        rows = [list(r) for r in rows]
        used = _all_diffs(rows)
        best_scope = _scope(rows)
        best_rows = [list(r) for r in rows]

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            current_scope = _scope(rows)

            # Pick a row and position to try to reduce
            # Bias toward positions with large values
            ri = rng.randint(0, n_rows - 1)
            row = rows[ri]

            # Prioritize last position if it contributes to scope
            if row[-1] == current_scope and rng.random() < 0.6:
                pi = k
            else:
                pi = rng.randint(1, k)

            old_val = row[pi]
            if old_val <= 1:
                continue

            lo = row[pi - 1] + 1
            hi = old_val - 1
            if pi < k:
                hi = min(hi, row[pi + 1] - 1)

            if lo > hi:
                continue

            # Try values from lo upward (prefer smallest)
            found = False
            for new_val in range(lo, hi + 1):
                # Compute old and new differences involving position pi
                old_diffs = set()
                for i in range(pi):
                    old_diffs.add(old_val - row[i])
                for i in range(pi + 1, len(row)):
                    old_diffs.add(row[i] - old_val)

                new_diffs = set()
                for i in range(pi):
                    new_diffs.add(new_val - row[i])
                for i in range(pi + 1, len(row)):
                    new_diffs.add(row[i] - new_val)

                other_diffs = used - old_diffs
                if new_diffs & other_diffs:
                    continue

                # Valid move
                row[pi] = new_val
                used -= old_diffs
                used |= new_diffs
                found = True

                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                break

            if not found:
                continue

        return best_rows, best_scope

    def _simulated_annealing(rows, deadline, rng, max_iter=50000):
        """Simulated annealing to minimize scope."""
        current = [list(r) for r in rows]
        current_scope = _scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope
        used = _all_diffs(current)

        temp = max(current_scope * 0.15, 8.0)
        cooling = 0.99985
        min_temp = 0.01

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if temp < min_temp:
                break

            ri = rng.randint(0, n_rows - 1)
            pi = rng.randint(1, k)

            row = current[ri]
            old_val = row[pi]

            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else current_scope

            if lo >= hi:
                continue

            # Try a random new value in range
            if rng.random() < 0.5:
                new_val = rng.randint(lo, hi)
            else:
                # Try to reduce
                if old_val > lo:
                    new_val = rng.randint(lo, old_val - 1)
                else:
                    continue

            if new_val == old_val:
                continue

            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - row[i])
            for i in range(pi + 1, len(row)):
                old_diffs.add(row[i] - old_val)

            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - row[i])
            for i in range(pi + 1, len(row)):
                new_diffs.add(row[i] - new_val)

            other_diffs = used - old_diffs
            if new_diffs & other_diffs:
                continue

            row[pi] = new_val
            used -= old_diffs
            used |= new_diffs

            new_scope = _scope(current)

            if new_scope <= current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
            else:
                delta = new_scope - current_scope
                prob = 1.0 - min(1.0, delta / (temp + 1e-9))
                if rng.random() > prob:
                    row[pi] = old_val
                    used -= new_diffs
                    used |= old_diffs
                else:
                    current_scope = new_scope

            temp *= cooling

        return best, best_scope

    def _backtrack_reduce_max(rows, target_scope, deadline, rng):
        """
        Try to reduce the maximum entry to target_scope using backtracking
        on the rows that currently exceed target_scope.
        """
        rows = [list(r) for r in rows]
        used = _all_diffs(rows)

        # Find rows that need reduction
        need_reduce = [i for i in range(n_rows) if rows[i][-1] > target_scope]
        if not need_reduce:
            return rows, True

        # Try to fix each problematic row
        for ri in need_reduce:
            if time.time() > deadline:
                return None, False

            row = rows[ri]
            # Backtrack from the end
            def _fix_row(row, pi, used_local, deadline_local):
                if pi < 0:
                    return True

                if time.time() > deadline_local:
                    return False

                lo = row[pi - 1] + 1 if pi > 0 else 1
                hi = min(target_scope, row[pi + 1] - 1) if pi < k else target_scope

                if lo > hi:
                    return False

                # Try values, preferring smaller ones
                old_val = row[pi]
                candidates = list(range(lo, hi + 1))
                # Put old_val first for stability, then shuffle rest
                if old_val in candidates:
                    candidates.remove(old_val)
                    candidates.insert(0, old_val)
                else:
                    rng.shuffle(candidates)

                for new_val in candidates:
                    if new_val == old_val:
                        continue

                    old_diffs = set()
                    for i in range(pi):
                        old_diffs.add(old_val - row[i])
                    for i in range(pi + 1, len(row)):
                        old_diffs.add(row[i] - old_val)

                    new_diffs = set()
                    for i in range(pi):
                        new_diffs.add(new_val - row[i])
                    for i in range(pi + 1, len(row)):
                        new_diffs.add(row[i] - new_val)

                    other_diffs = used_local - old_diffs
                    if new_diffs & other_diffs:
                        continue

                    row[pi] = new_val
                    used_local_new = used_local - old_diffs | new_diffs

                    if _fix_row(row, pi - 1, used_local_new, deadline_local):
                        used_local.clear()
                        used_local.update(used_local_new)
                        return True

                    row[pi] = old_val

                return False

            if not _fix_row(row, k, used, deadline - 2):
                return None, False

            # Update global used
            used = _all_diffs(rows)

        return rows, True

    # Phase 1: Generate diverse initial solutions using column-wise construction
    best_rows = None
    best_scope = float('inf')

    for seed in range(15):
        if time.time() > deadline - 50:
            break
        rows = _col_wise_construct(seed, use_randomization=True)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

    # Also try deterministic column-wise
    for seed in range(5):
        if time.time() > deadline - 50:
            break
        rows = _col_wise_construct(seed, use_randomization=False)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

    if best_rows is None:
        # Ultimate fallback: simple greedy
        rows = _col_wise_construct(0, use_randomization=False)
        if rows is None:
            # Manual construction
            rows = [[0, 1, 2, 3, 4, 5, 6]]
            for i in range(1, n_rows):
                row = [0]
                for j in range(1, k + 1):
                    x = j + i * (k + 1)
                    row.append(x)
                rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 2: Local search to shrink
    rng = random.Random(123)
    if time.time() < deadline - 35:
        polished, ps_scope = _element_local_search(best_rows, deadline - 30, rng, max_iter=20000)
        if ps_scope < best_scope:
            best_rows = polished
            best_scope = ps_scope

    # Phase 3: Simulated annealing
    if time.time() < deadline - 25:
        sa_rows, sa_scope = _simulated_annealing(best_rows, deadline - 20, rng, max_iter=40000)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Phase 4: Try to reduce scope via backtracking on specific target
    current_max = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)

    # Try progressively lower targets
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        target = current_max - 1
        if target < lower_bound:
            break

        rng2 = random.Random(1000 + attempt)
        result, success = _backtrack_reduce_max(best_rows, target, deadline - 10, rng2)
        if success and result is not None:
            new_scope = _scope(result)
            if new_scope < best_scope:
                best_rows = result
                best_scope = new_scope
                current_max = new_scope

    # Phase 5: Final polish
    if time.time() < deadline - 5:
        rng3 = random.Random(999)
        polished, ps_scope = _element_local_search(best_rows, deadline - 3, rng3, max_iter=15000)
        if ps_scope < best_scope:
            best_rows = polished
            best_scope = ps_scope

    # Final validation
    if not _valid(best_rows):
        best_rows = _col_wise_construct(0, use_randomization=False)
        if best_rows is None:
            rows = [[0, 1, 2, 3, 4, 5, 6]]
            for i in range(1, n_rows):
                row = [0]
                for j in range(1, k + 1):
                    x = j + i * (k + 1)
                    row.append(x)
                rows.append(row)
            best_rows = rows

    return best_rows
