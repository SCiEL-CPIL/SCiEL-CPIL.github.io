"""Rewritten program: structural change to a randomized local-search / simulated-annealing
approach over the difference-set representation, with a hard wall-clock deadline guard."""
import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    # Simple counting bound: largest entry m must satisfy m(m+1)/2 >= total_diffs
    m_min = 0
    while m_min * (m_min + 1) // 2 < total_diffs:
        m_min += 1
    m_min -= 1  # the smallest m with m(m+1)/2 >= total_diffs is m_min+1; m_min is just below
    # Actually m_min is now the largest m with m(m+1)/2 < total_diffs, so lower bound is m_min+1
    lower_bound = m_min + 1

    best_rows = None
    best_scope = None

    def validate(rows):
        """Return True if all positive differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return False
                    seen.add(diff)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def rows_from_diffs(diff_list, row_len):
        """Given a flat list of differences (length n_rows*d), try to reconstruct rows.
        We assume each row's differences are stored in order, and we reconstruct by
        greedily placing marks. This is only valid if the diffs are consistent with a ruler.
        For local search we instead work directly on rows."""
        pass

    # We'll work directly on rows. Represent a candidate as list of rows.
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

    # Simulated annealing / randomized local search
    # Move: pick a random row, pick a random mark position (not 0), change its value
    # or swap two marks, or shift a suffix.
    # We'll use a simple approach: pick a random row and a random non-zero mark,
    # try to move it to a random new value, check validity incrementally.

    rng = random.Random(42)

    def row_diffs(row):
        """Return sorted list of positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def all_diffs_set(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    current_diffs = all_diffs_set(rows)
    current_scope = best_scope

    def try_move(rows, diff_set, row_idx, pos, new_val):
        """Attempt to set rows[row_idx][pos] = new_val. Returns (success, new_diff_set, new_scope)."""
        row = rows[row_idx]
        if pos == 0:
            return False, diff_set, get_scope(rows)
        if new_val <= row[pos - 1] or new_val >= row[pos + 1] if pos < len(row) - 1 else new_val <= row[pos - 1]:
            return False, diff_set, get_scope(rows)
        # Also must be > 0
        if new_val <= 0:
            return False, diff_set, get_scope(rows)

        old_val = row[pos]
        # Compute diffs that involve position pos
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)

        # Check if new diffs conflict with (diff_set - old_diffs)
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)

        # The diffs that remain from this row without pos
        other_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if i == pos or j == pos:
                    continue
                other_diffs.add(row[j] - row[i])

        # Diffs from other rows
        other_rows_diffs = set()
        for ri in range(n_rows):
            if ri == row_idx:
                continue
            r = rows[ri]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    other_rows_diffs.add(r[j] - r[i])

        # New total diff set would be: other_diffs | new_diffs | other_rows_diffs
        # Must have no duplicates: len(new_diffs) == d-? Actually new_diffs has k elements (diffs to other k marks)
        # other_diffs has d - k elements
        # Check new_diffs doesn't intersect with (other_diffs | other_rows_diffs)
        combined_other = other_diffs | other_rows_diffs
        if new_diffs & combined_other:
            return False, diff_set, get_scope(rows)
        # Also new_diffs must be all distinct among themselves (they are, since row is strictly increasing)
        # And other_diffs must be distinct (they are)
        # And other_rows_diffs distinct (they are)
        
        new_row = row[:]
        new_row[pos] = new_val
        # Ensure strictly increasing
        for i in range(len(new_row) - 1):
            if new_row[i] >= new_row[i + 1]:
                return False, diff_set, get_scope(rows)

        new_scope = max(max(r) for r in rows[:row_idx] + [new_row] + rows[row_idx + 1:])
        new_diff_set = combined_other | new_diffs
        return True, new_diff_set, new_scope

    # Better: just do full validation for simplicity, but that's O(n_rows * k^2) per move.
    # With n_rows=4, k=7, k+1=8, d=28, total_diffs=112. Validation is ~112 set ops, fast.

    def full_validate_scope(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return None
                    seen.add(diff)
        return max(max(r) for r in rows)

    # Local search with restarts
    max_restarts = 200
    steps_per_restart = 50000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or random perturbation
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: random row, random mark, random new value
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                old = cur[ri][pos]
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Annealing
        T = max(0.1, 1.0 - restart / max_restarts)
        for step in range(steps_per_restart):
            if time.time() > deadline:
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
            new_scope = full_validate_scope(cur)

            if new_scope is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                # Accept if better or with probability exp(-delta/T)
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                # Invalid: reject
                cur[ri][pos] = old_val

            T *= 0.9999

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
