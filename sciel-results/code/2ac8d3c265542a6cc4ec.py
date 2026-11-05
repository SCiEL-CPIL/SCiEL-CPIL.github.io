import random
import time
import sys
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = 115.0  # Safe margin before 120s kill

    def get_scope(rows):
        return max(row[-1] for row in rows)

    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == n_rows * k * (k + 1) // 2

    def greedy_construct(n_rows, k, max_scope=200):
        """Build rows greedily, one at a time."""
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                for x in range(row[-1] + 1, max_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Try with larger scope
                    for x in range(row[-1] + 1, max_scope + 50):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                if not found:
                    return None
            rows.append(row)
        return rows

    def simulated_annealing(rows, time_budget=100.0):
        """SA to minimize the scope while maintaining validity."""
        if rows is None:
            return None

        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)

        # Track used differences for quick validity check
        current_diffs = get_all_diffs(current_rows)

        # SA parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995
        max_iter = 1000000

        rng = random.Random(42)

        def try_move(row_idx, pos_idx, new_val):
            """Try to change current_rows[row_idx][pos_idx] to new_val.
            Returns (new_diffs_set, new_scope) if valid, else None."""
            row = current_rows[row_idx]
            old_val = row[pos_idx]
            if new_val <= old_val or (pos_idx + 1 < len(row) and new_val >= row[pos_idx + 1]):
                return None
            if pos_idx > 0 and new_val <= row[pos_idx - 1]:
                return None

            # Compute new diffs for this row
            old_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_row_diffs.add(row[j] - row[i])

            # Remove old diffs, add new diffs
            temp_row = row[:]
            temp_row[pos_idx] = new_val
            new_row_diffs = set()
            for i in range(len(temp_row)):
                for j in range(i + 1, len(temp_row)):
                    new_row_diffs.add(temp_row[j] - temp_row[i])

            # Check for conflicts
            # Diffs that are in old_row_diffs but not in new_row_diffs can be freed
            # Diffs that are in new_row_diffs but not in old_row_diffs must not conflict
            freed = old_row_diffs - new_row_diffs
            added = new_row_diffs - old_row_diffs

            # Check if any added diff is in current_diffs (excluding freed ones)
            for d in added:
                if d in current_diffs and d not in freed:
                    return None

            new_diffs = (current_diffs - freed) | added
            new_scope = max(new_val, get_scope(current_rows))
            # Actually recompute scope properly
            new_rows = [r[:] for r in current_rows]
            new_rows[row_idx][pos_idx] = new_val
            new_scope = get_scope(new_rows)

            return (new_diffs, new_scope)

        def do_move(row_idx, pos_idx, new_val, new_diffs, new_scope):
            """Apply a move."""
            current_rows[row_idx][pos_idx] = new_val

        iter_count = 0
        while time.time() - start_time < deadline - 5.0 and iter_count < max_iter:
            iter_count += 1

            # Randomly pick a row and position to modify
            row_idx = rng.randint(0, n_rows - 1)
            pos_idx = rng.randint(1, k)  # positions 1..k (position 0 is always 0)

            row = current_rows[row_idx]
            # Generate a new value
            if pos_idx == 1:
                lo, hi = 1, row[1] + 5 if row[1] > 1 else 10
            else:
                lo = row[pos_idx - 1] + 1
                hi = row[pos_idx] + 5 if pos_idx < k else row[pos_idx] + 10

            if lo > hi:
                hi = lo + 10

            new_val = rng.randint(lo, hi)

            if new_val == row[pos_idx]:
                continue

            result = try_move(row_idx, pos_idx, new_val)
            if result is None:
                continue

            new_diffs, new_scope = result

            delta_scope = new_scope - current_scope

            if delta_scope <= 0:
                # Accept
                do_move(row_idx, pos_idx, new_val, new_diffs, new_scope)
                current_diffs = new_diffs
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Maybe accept with probability exp(-delta/T)
                if T > T_min and rng.random() < math.exp(-delta_scope / T):
                    do_move(row_idx, pos_idx, new_val, new_diffs, new_scope)
                    current_diffs = new_diffs
                    current_scope = new_scope

            T = max(T * cooling_rate, T_min)

        return best_rows

    def polish_max(rows):
        """Try to reduce the maximum entry by adjusting individual elements."""
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        rng = random.Random(123)

        for _ in range(5000):
            if time.time() - start_time > deadline - 5.0:
                break

            # Find the row with the max entry
            max_val = best_scope
            max_row_idx = None
            for i, row in enumerate(best):
                if row[-1] == max_val:
                    max_row_idx = i
                    break

            if max_row_idx is None:
                break

            row = best[max_row_idx]

            # Try to reduce the last element
            pos = len(row) - 1
            old_val = row[pos]
            # Try values from old_val - 1 down to row[pos-1] + 1
            for new_val in range(old_val - 1, row[pos - 1], -1):
                # Check validity
                temp_row = row[:]
                temp_row[pos] = new_val
                # Check increasing
                if any(temp_row[i] >= temp_row[i + 1] for i in range(len(temp_row) - 1)):
                    continue

                # Check diff conflicts
                all_diffs = set()
                for r_idx, r in enumerate(best):
                    if r_idx == max_row_idx:
                        r = temp_row
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            all_diffs.add(r[j] - r[i])

                if len(all_diffs) == n_rows * k * (k + 1) // 2:
                    # Valid
                    best[max_row_idx] = temp_row
                    best_scope = new_val
                    break

        return best

    # Phase 1: Greedy initialization
    rows = greedy_construct(n_rows, k, max_scope=150)
    if rows is None:
        rows = greedy_construct(n_rows, k, max_scope=200)
    if rows is None:
        rows = greedy_construct(n_rows, k, max_scope=300)
    if rows is None:
        # Fallback: very large scope
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)

    if not is_valid(rows):
        # Should not happen with greedy, but safety check
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)

    best_result = rows
    best_scope = get_scope(best_result)

    # Phase 2: Simulated annealing with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 20.0:
            break

        # For restarts after the first, perturb the current best
        if restart == 0:
            sa_rows = [r[:] for r in rows]
        else:
            # Perturb: randomly change some elements
            sa_rows = [r[:] for r in best_result]
            rng = random.Random(42 + restart)
            for _ in range(10):
                ri = rng.randint(0, n_rows - 1)
                pi = rng.randint(1, k)
                old = sa_rows[ri][pi]
                # Try a different value
                lo = sa_rows[ri][pi - 1] + 1 if pi > 0 else 1
                hi = old + 10
                new_val = rng.randint(lo, hi)
                if new_val != old:
                    sa_rows[ri][pi] = new_val
            # Check validity after perturbation
            if not is_valid(sa_rows):
                sa_rows = [r[:] for r in best_result]

        sa_result = simulated_annealing(sa_rows, time_budget=100.0)
        if sa_result is not None and is_valid(sa_result):
            sa_scope = get_scope(sa_result)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_result = sa_result

        # Phase 3: Polish
        if time.time() - start_time > deadline - 10.0:
            break
        polished = polish_max(best_result)
        if is_valid(polished):
            pol_scope = get_scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_result = polished

    return best_result
