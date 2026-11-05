"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    so that all positive differences a_j - a_i (i<j) across all rows are distinct.
    Minimize the largest entry (scope).
    """
    t_start = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 15.0

    # Total differences per row: C(k+1, 2) = k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Bitmask approach: we track which differences (1..max_possible) are used.
    # We'll use a Python int as a bitset where bit d is set if difference d is used.

    def row_diffs(row):
        """Compute set of differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def row_diff_bitset(row):
        """Compute bitset of differences for a row."""
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                bits |= (1 << d)
        return bits

    # Greedy construction to get an initial solution
    def greedy_construct():
        used_bits = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if adding x creates any conflict
                conflict = False
                for a in row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    # Also check internal conflicts (differences within new row)
                    # Since row is increasing and we only add x at the end,
                    # new diffs are x - a for a in row, which we already checked against used_bits
                    # But we need to ensure no two of these new diffs are equal to each other
                    # They can't be equal since x - a1 != x - a2 if a1 != a2
                    row.append(x)
                    for a in row[:-1]:
                        used_bits |= (1 << (x - a))
            rows.append(row)
        return rows

    # Try to get a good initial solution
    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)

    # Simulated annealing to improve
    def sa_improve(rows, max_time=10.0):
        """Improve rows via simulated annealing."""
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)

        # Compute used differences
        used_bits = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_bits |= (1 << d)

        t = time.time()
        deadline = t_start + DEADLINE - SAFETY_MARGIN
        temp = scope * 0.5
        cooling = 0.999
        best_local = scope
        best_rows_local = [list(r) for r in rows]

        while temp > 0.1 and time.time() < deadline:
            # Pick a random row and element to move
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, len(rows[ri]) - 1)  # Don't move the first element (must be 0)
            old_val = rows[ri][ei]

            # Try a random new value
            if random.random() < 0.5:
                # Small move
                new_val = old_val + random.choice([-2, -1, 1, 2])
            else:
                # Larger move
                new_val = old_val + random.randint(-5, 5)

            if new_val <= 0 or new_val >= rows[ri][ei + 1] if ei < len(rows[ri]) - 1 else new_val >= rows[ri][-1]:
                # Check bounds: must be greater than previous and less than next
                if ei > 0 and new_val <= rows[ri][ei - 1]:
                    continue
                if ei < len(rows[ri]) - 1 and new_val >= rows[ri][ei + 1]:
                    continue
                if new_val <= 0:
                    continue

            # Compute old differences for this element
            old_diffs = []
            for j in range(len(rows[ri])):
                if j != ei:
                    d = abs(rows[ri][j] - old_val)
                    old_diffs.append((j, d))

            # Compute new differences
            new_diffs = []
            for j in range(len(rows[ri])):
                if j != ei:
                    d = abs(rows[ri][j] - new_val)
                    new_diffs.append((j, d))

            # Check if new diffs conflict with used_bits (excluding old diffs)
            # Remove old diffs from used_bits temporarily
            temp_bits = used_bits
            for (j, d) in old_diffs:
                temp_bits &= ~(1 << d)

            # Check new diffs against temp_bits
            valid = True
            for (j, d) in new_diffs:
                if (temp_bits >> d) & 1:
                    valid = False
                    break
                # Also check if any two new diffs are the same (they shouldn't be since positions are distinct)

            if not valid:
                # Revert
                continue

            # Compute new scope
            new_max = max(rows[ri][ei - 1] if ei > 0 else 0, new_val,
                         rows[ri][ei + 1] if ei < len(rows[ri]) - 1 else 0)
            # Actually need to recompute scope properly
            new_scope = scope
            if new_val > scope:
                new_scope = new_val
            # Check if old_val was the max and we're reducing it
            # This is complex; let's just compute it
            new_scope = 0
            for r in rows:
                if max(r) > new_scope:
                    new_scope = max(r)
            # But we haven't updated rows[ri][ei] yet
            # Let's just compute the potential new scope
            row_max = max(rows[ri][:ei] + [new_val] + rows[ri][ei+1:])
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    if max(rows[i]) > other_max:
                        other_max = max(rows[i])
            new_scope = max(row_max, other_max)

            # Acceptance criterion
            if new_scope <= scope:
                # Always accept improvements
                pass
            else:
                # Accept worse with probability exp(-delta/temp)
                delta = new_scope - scope
                if random.random() > math.exp(-delta / temp):
                    continue

            # Apply move
            # Remove old diffs from used_bits
            for (j, d) in old_diffs:
                used_bits &= ~(1 << d)
            rows[ri][ei] = new_val
            for (j, d) in new_diffs:
                used_bits |= (1 << d)
            scope = new_scope

            if scope < best_local:
                best_local = scope
                best_rows_local = [list(r) for r in rows]

            temp *= cooling

        return best_rows_local, best_local

    # Run SA with time limit
    sa_deadline = t_start + DEADLINE - SAFETY_MARGIN
    while time.time() < sa_deadline - 5.0:
        improved_rows, improved_scope = sa_improve(best_rows, max_time=5.0)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        else:
            # Try a fresh restart from greedy
            if time.time() > sa_deadline - 10.0:
                break
            best_rows = greedy_construct()
            best_scope = max(max(r) for r in best_rows)

    # Final micro-polish: try to reduce the maximum element
    if time.time() < sa_deadline - 2.0:
        # Find the row with the maximum element
        max_val = best_scope
        max_row_idx = None
        max_elem_idx = None
        for i, row in enumerate(best_rows):
            if row[-1] == max_val:
                max_row_idx = i
                max_elem_idx = len(row) - 1
                break

        if max_row_idx is not None:
            # Try to reduce the last element
            row = best_rows[max_row_idx]
            # Compute used bits
            used_bits = 0
            for r in best_rows:
                for i2 in range(len(r)):
                    for j2 in range(i2 + 1, len(r)):
                        d = r[j2] - r[i2]
                        used_bits |= (1 << d)

            # Remove diffs involving the last element
            for j in range(len(row) - 1):
                d = row[-1] - row[j]
                used_bits &= ~(1 << d)

            # Try to find a smaller value for the last element
            for new_val in range(row[-1] - 1, row[-2], -1):
                valid = True
                for j in range(len(row) - 1):
                    d = new_val - row[j]
                    if (used_bits >> d) & 1:
                        valid = False
                        break
                if valid:
                    # Apply
                    row[-1] = new_val
                    for j in range(len(row) - 1):
                        d = new_val - row[j]
                        used_bits |= (1 << d)
                    best_scope = max(max(r) for r in best_rows)
                    break

    return best_rows
# EVOLVE-BLOCK-END
