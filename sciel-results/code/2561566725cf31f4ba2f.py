"""Rewritten program: targeted-descent local search that directly minimizes the global
scope (largest entry) by attempting to reduce the largest elements, with efficient
conflict checking and a hard wall-clock deadline guard."""
import time
import random
import sys
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 140.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    # Simple counting bound: largest entry m must satisfy m(m+1)/2 >= total_diffs
    m_min = 0
    while m_min * (m_min + 1) // 2 < total_diffs:
        m_min += 1
    lower_bound = m_min

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Initial solution: greedy
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

    rng = random.Random(42)

    # Fast validation: returns (valid, scope) using early termination
    def fast_validate(rows):
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                if ri > scope:
                    scope = ri
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return (False, None)
                    seen.add(diff)
        return (True, scope)

    # Precompute all differences for a set of rows
    def compute_all_diffs(rows):
        """Return a set of all differences across all rows."""
        diffs = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.add(row[j] - row[i])
        return diffs

    # Check if a single row's differences conflict with a set of other diffs
    def row_diffs_conflict(row, other_diffs):
        """Check if any difference in row conflicts with other_diffs."""
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                if row[j] - row[i] in other_diffs:
                    return True
        return False

    # Get the row and position of the maximum element
    def find_max_element(rows):
        max_val = 0
        max_row = 0
        max_pos = 0
        for ri, row in enumerate(rows):
            for pi, val in enumerate(row):
                if val > max_val:
                    max_val = val
                    max_row = ri
                    max_pos = pi
        return max_row, max_pos, max_val

    # Targeted descent: try to reduce the largest element
    def targeted_descent(rows, scope, max_iters=100000):
        """Try to reduce the scope by lowering the largest elements."""
        rows = [r[:] for r in rows]
        cur_scope = scope
        all_diffs = compute_all_diffs(rows)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Find the current maximum element
            max_row, max_pos, max_val = find_max_element(rows)
            if max_val <= cur_scope - 1:
                # Already reduced
                pass

            # Try to reduce the maximum element
            row = rows[max_row]
            old_val = row[max_pos]

            # Determine valid range for new value
            lo = row[max_pos - 1] + 1 if max_pos > 0 else 1
            hi = row[max_pos + 1] - 1 if max_pos < len(row) - 1 else old_val - 1

            if hi < lo:
                # Can't reduce this element, try next largest
                # Mark as temporarily increased to skip
                continue

            # Try values from hi down to lo
            found = False
            for new_val in range(hi, lo - 1, -1):
                if new_val == old_val:
                    continue

                # Check if this value creates conflicts
                # Remove old differences from all_diffs
                old_diffs = set()
                n = len(row)
                for i in range(n):
                    if i != max_pos:
                        old_diffs.add(row[max_pos] - row[i] if row[max_pos] > row[i] else row[i] - row[max_pos])

                # Check new differences
                new_diffs = set()
                for i in range(n):
                    if i != max_pos:
                        new_diffs.add(new_val - row[i] if new_val > row[i] else row[i] - new_val)

                # Check if new_diffs conflict with remaining diffs
                remaining = all_diffs - old_diffs
                if new_diffs & remaining:
                    continue

                # No conflict! Update
                row[max_pos] = new_val
                all_diffs = remaining | new_diffs
                cur_scope = max(max(r) for r in rows)
                found = True
                break

            if not found:
                # Try a random perturbation to escape local minimum
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(rows[ri]) - 1)
                old = rows[ri][pos]
                lo_p = rows[ri][pos - 1] + 1
                hi_p = rows[ri][pos + 1] - 1 if pos < len(rows[ri]) - 1 else cur_scope + 10
                if hi_p > lo_p:
                    new_p = rng.randint(lo_p, hi_p)
                    if new_p != old:
                        # Check validity
                        rows[ri][pos] = new_p
                        valid, _ = fast_validate(rows)
                        if valid:
                            all_diffs = compute_all_diffs(rows)
                            cur_scope = max(max(r) for r in rows)
                        else:
                            rows[ri][pos] = old

            # Check if we've reached the lower bound
            if cur_scope <= lower_bound:
                break

        return rows, cur_scope

    # Main search loop
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            cur = [r[:] for r in best_rows]
            cur_scope = best_scope
        else:
            # Start from best and perturb
            cur = [r[:] for r in best_rows]
            cur_scope = best_scope
            # Perturb a few elements
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                old = cur[ri][pos]
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

            valid, cur_scope = fast_validate(cur)
            if not valid:
                continue

        # Apply targeted descent
        cur, cur_scope = targeted_descent(cur, cur_scope)

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # If we've hit the lower bound, we're done
        if best_scope <= lower_bound:
            break

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
