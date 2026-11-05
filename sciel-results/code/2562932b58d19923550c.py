"""
Difference triangle set (4,7): CSP with bitset difference propagation.
Strategy:
  1. Fast greedy initialization.
  2. Local search (row-rebuild) to reduce max entry.
  3. Exact CSP solver with bitset-based difference tracking,
     dynamic row ordering by tightness, and value pruning.
     Searches for the smallest M that admits a valid solution.
"""
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 145.0

    def all_diffs(rows):
        diffs = []
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.append(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    # ---------- Greedy initialization ----------
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

    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # ---------- Local search: rebuild rows to reduce max ----------
    def rebuild_row(forbidden_diffs, rng, max_val_limit):
        """Build a row of k+1 increasing ints starting at 0,
        avoiding forbidden_diffs, with max entry <= max_val_limit."""
        row = [0]
        local_used = set()
        for _ in range(k):
            last = row[-1]
            candidates = []
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    candidates.append(x)
                    if len(candidates) >= 8:
                        break
            if not candidates:
                return None
            # Bias towards smaller values, but allow randomness
            if rng.random() < 0.4:
                x = candidates[0]
            else:
                x = candidates[rng.randint(0, len(candidates) - 1)]
            row.append(x)
            local_used.update({x - a for a in row[:-1]})
        return row

    rng = random.Random(42)
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    used_diffs = set()
    for r in current_rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used_diffs.add(r[j] - r[i])

    no_improve = 0
    while time.time() < deadline - 50.0:
        i = rng.randint(0, n_rows - 1)
        old_row = current_rows[i]
        old_diffs = set()
        n = len(old_row)
        for a in range(n):
            for b in range(a + 1, n):
                old_diffs.add(old_row[b] - old_row[a])
        used_diffs.difference_update(old_diffs)

        # Try to improve: max < current_max
        target = current_max - 1
        new_row = None
        if target >= k:
            new_row = rebuild_row(used_diffs, rng, target)
        if new_row is None:
            # Try neutral: max <= current_max
            new_row = rebuild_row(used_diffs, rng, current_max)
        if new_row is None:
            # Restore
            current_rows[i] = old_row
            used_diffs.update(old_diffs)
            no_improve += 1
            if no_improve > 200:
                break
            continue

        current_rows[i] = new_row
        new_diffs = set()
        n2 = len(new_row)
        for a in range(n2):
            for b in range(a + 1, n2):
                new_diffs.add(new_row[b] - new_row[a])
        used_diffs.update(new_diffs)

        new_max = max_entry(current_rows)
        if new_max < best_max:
            best_max = new_max
            best_rows = [rr[:] for rr in current_rows]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 200:
                break

    # ---------- Exact CSP solver with bitset propagation ----------
    total_diffs = n_rows * k * (k + 1) // 2

    def solve_for_M(M):
        """
        Exact backtracking with bitset-based difference tracking.
        Returns a valid list of rows if found within time, else None.
        """
        start_time = time.time()
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        # used_diffs_bitset: bit i set means difference (i+1) is used
        # We need differences up to M, so we need M bits.
        used_bits = 0  # Python int as bitset

        def diff_bitset(row, upto):
            """Return bitset of all differences in row[0..upto] (inclusive)."""
            bits = 0
            for i in range(upto + 1):
                for j in range(i + 1, upto + 1):
                    d = row[j] - row[i]
                    if 1 <= d <= M:
                        bits |= (1 << d)
            return bits

        def remaining_capacity(r):
            """
            Lower bound on the max entry needed for row r given current assignments.
            Row r has positions 0..k. Positions already assigned: those < current_pos[r].
            The remaining positions must be filled with increasing values.
            The minimum possible max entry is at least the last assigned value + remaining_count.
            """
            # Simple heuristic: last assigned + number of remaining positions
            last_val = rows[r][current_pos[r] - 1] if current_pos[r] > 0 else 0
            remaining = k + 1 - current_pos[r]
            return last_val + remaining

        # Dynamic row ordering: fill the row with the tightest remaining capacity first
        # We'll use a priority approach: at each step, pick the row with the smallest
        # "slack" (M - current_max_of_row - remaining_positions)

        current_pos = [0] * n_rows  # next position to fill for each row (0-indexed, 0 means position 0 is done)

        def order_rows():
            """Return rows ordered by tightness (smallest slack first)."""
            slacks = []
            for r in range(n_rows):
                if current_pos[r] >= k + 1:
                    slacks.append((999999, r))
                else:
                    last_val = rows[r][current_pos[r] - 1] if current_pos[r] > 0 else 0
                    remaining = k + 1 - current_pos[r]
                    slack = M - (last_val + remaining)
                    slacks.append((slack, r))
            slacks.sort()
            return [r for _, r in slacks]

        def backtrack():
            nonlocal used_bits
            if time.time() - start_time > 8.0:
                return False

            # Find the next row to assign
            active = [r for r in range(n_rows) if current_pos[r] < k + 1]
            if not active:
                return True

            # Order by tightness
            active.sort(key=lambda r: (M - (rows[r][current_pos[r] - 1] if current_pos[r] > 0 else 0) - (k + 1 - current_pos[r])))
            r = active[0]
            p = current_pos[r]

            # Try values for position p
            lower = rows[r][p - 1] + 1 if p > 0 else 1
            upper = M

            # Prune: the remaining positions after p need at least (k - p) more values
            # each at least 1 apart, so rows[r][p] <= M - (k - p)
            upper = min(upper, M - (k - p))

            for v in range(lower, upper + 1):
                # Check differences with already-assigned positions in this row
                new_diffs = []
                ok = True
                for i in range(p):
                    d = v - rows[r][i]
                    if d <= 0 or d > M:
                        ok = False
                        break
                    if (used_bits >> d) & 1:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue

                # Check within-row uniqueness of new diffs
                if len(new_diffs) != len(set(new_diffs)):
                    continue

                # Also check: new diffs must not conflict with each other
                # (already checked above)

                # Assign
                rows[r][p] = v
                new_bits = 0
                for d in new_diffs:
                    new_bits |= (1 << d)
                used_bits |= new_bits

                current_pos[r] = p + 1

                if backtrack():
                    return True

                # Unassign
                used_bits ^= new_bits
                current_pos[r] = p

            return False

        if backtrack():
            return [row[:] for row in rows]
        return None

    # ---------- Search for smallest M ----------
    # Start from current best_max and go down
    for M in range(best_max - 1, max(total_diffs, best_max - 25) - 1, -1):
        if time.time() > deadline - 5.0:
            break
        sol = solve_for_M(M)
        if sol is not None:
            diffs = all_diffs(sol)
            if len(diffs) == len(set(diffs)):
                best_rows = sol
                best_max = M
                break

    return best_rows
