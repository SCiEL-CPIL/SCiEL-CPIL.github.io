import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 105  # 150s wall clock limit, leave 45s margin

    def elapsed():
        return time.time() - start_time

    def dead():
        return elapsed() > deadline

    def diffs_of_row(row):
        """Return the set of all positive differences in a row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def scope(rows):
        """Largest entry across all rows."""
        return max(max(r) for r in rows)

    def valid(rows):
        """Check all positive differences are distinct across all rows."""
        all_diffs = []
        for r in rows:
            all_diffs.extend(diffs_of_row(r))
        return len(all_diffs) == len(set(all_diffs))

    def greedy_row(used, max_val):
        """Greedy: build one row of k+1 elements starting at 0."""
        row = [0]
        for _ in range(k):
            # Find smallest x > row[-1] such that all new diffs are unused
            x = row[-1] + 1
            while x <= max_val:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
            else:
                return None
        return row

    def greedy_construct(n, k, max_val):
        """Build n rows greedily."""
        used = set()
        rows = []
        for _ in range(n):
            row = greedy_row(used, max_val)
            if row is None:
                return None
            rows.append(row)
        return rows

    # First, try greedy with various max_val to get a starting solution
    best_rows = None
    best_scope = float('inf')

    # Try greedy from low max_val upwards
    # Lower bound: need n_rows * k*(k+1)/2 distinct diffs, so max_val >= that count
    num_diffs = n_rows * (k * (k + 1) // 2)
    for M in range(num_diffs, 200):
        if dead():
            break
        rows = greedy_construct(n_rows, k, M)
        if rows is not None and valid(rows):
            s = scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
            if s == M:
                break  # Can't do better than this M with greedy

    if best_rows is None:
        # Fallback: just use a large M
        rows = greedy_construct(n_rows, k, 300)
        if rows is not None:
            return rows
        # Absolute fallback
        return [[0] + list(range(1, k + 1)) for _ in range(n_rows)]

    # Now try to improve with local search / reduction
    # Strategy: try to reduce the scope by replacing elements in rows
    # with smaller values that don't conflict

    def try_reduce(rows, max_iter=50000):
        """Try to reduce scope by finding smaller replacements."""
        rows = [r[:] for r in rows]
        current_scope = scope(rows)
        improved = True
        iters = 0

        while improved and not dead() and iters < max_iter:
            improved = False
            iters += 1

            # Recompute all used diffs
            all_diffs = set()
            for r in rows:
                all_diffs.update(diffs_of_row(r))

            # Try to reduce each element in each row
            for ri in range(len(rows)):
                if dead():
                    return rows
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):  # from end to start
                    if row[pos] <= current_scope - 1:
                        continue  # already small

                    old_val = row[pos]
                    # Try smaller values
                    for new_val in range(row[pos - 1] + 1, old_val):
                        # Check if new_val works
                        # Remove old diffs involving row[pos]
                        old_diffs = {old_val - row[i] for i in range(pos)}
                        # Add new diffs
                        new_diffs = {new_val - row[i] for i in range(pos)}

                        # Check no conflicts with other rows or remaining in this row
                        # Remove old_diffs from all_diffs, check new_diffs not in remaining
                        temp_diffs = set(all_diffs)
                        temp_diffs.difference_update(old_diffs)
                        if new_diffs & temp_diffs:
                            continue

                        # Also check new_val < row[pos+1] if exists
                        if pos < len(row) - 1 and new_val >= row[pos + 1]:
                            continue

                        # Accept
                        row[pos] = new_val
                        all_diffs = temp_diffs | new_diffs
                        current_scope = min(current_scope, max(max(r) for r in rows))
                        improved = True
                        break

        return rows

    # Apply reduction
    best_rows = try_reduce(best_rows)
    best_scope = scope(best_rows)

    # Try more aggressive local search: iterated perturbation
    random.seed(42)
    for iteration in range(50):
        if dead():
            break

        # Perturb: randomly change some elements
        perturbed = [r[:] for r in best_rows]
        num_changes = random.randint(1, 3)
        for _ in range(num_changes):
            ri = random.randint(0, len(perturbed) - 1)
            pos = random.randint(1, len(perturbed[ri]) - 1)
            # Increase the value (perturbation)
            perturbed[ri][pos] = min(perturbed[ri][pos] + random.randint(1, 10), 200)

        # Try to reduce
        reduced = try_reduce(perturbed, max_iter=10000)
        if reduced is not None and valid(reduced):
            s = scope(reduced)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in reduced]

    return best_rows
