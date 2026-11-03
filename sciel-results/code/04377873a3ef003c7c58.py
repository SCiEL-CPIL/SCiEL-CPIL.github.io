"""Difference triangle set solver: Simulated Annealing with bitmask diff tracking."""
import time
import sys
import random
import math

sys.setrecursionlimit(100000)


def _validate(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _sum_entries(rows):
    return sum(sum(r) for r in rows)


def _row_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _greedy(n_rows, k):
    """Fast greedy: smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            new = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new.append(d)
            if ok:
                row.append(x)
                used.update(new)
        rows.append(row)
    return rows


def _sa_optimize(n_rows, k, initial_rows, deadline, seed=42):
    """Simulated Annealing to minimize max entry, with sum as secondary cost."""
    rng = random.Random(seed)
    rows = [list(r) for r in initial_rows]
    n = k + 1  # entries per row
    total_diffs_per_row = n * (n - 1) // 2

    # Precompute difference index pairs for each position
    # For row of length n, position j (1..n-1) introduces diffs with positions 0..j-1
    # We track used diffs as a set of integers (the difference values)

    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def row_diffs_mask(row):
        """Return set of diffs for this row only."""
        d = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d.add(row[j] - row[i])
        return d

    def is_valid_state(rows):
        """Check if all rows have distinct differences from each other."""
        all_diffs = set()
        for row in rows:
            rd = row_diffs_mask(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True

    def cost(rows):
        """Primary: max entry. Secondary: sum of entries."""
        m = max(max(r) for r in rows)
        s = sum(sum(r) for r in rows)
        return m * 10000 + s

    # Initial state
    cur_rows = [list(r) for r in rows]
    cur_cost = cost(cur_rows)
    best_rows = [list(r) for r in cur_rows]
    best_cost = cur_cost

    # SA parameters
    t = 1.0
    t_min = 0.01
    cooling = 0.9998  # slow cooling

    # Precompute: for a given row, which differences are "internal"
    # A move: pick a row, pick a position (1..n-1), change its value
    # The new value must be > previous and < next (if exists)
    # And all new diffs must not collide with other rows' diffs or other internal diffs

    # Efficient diff tracking: maintain a global used_diffs set
    # When we modify row r at position p, we need to:
    # 1. Remove old diffs involving position p from global
    # 2. Check new diffs don't collide with remaining global
    # 3. Add new diffs to global

    # Let's use a different approach: maintain per-row diff sets and a global count
    # Actually, for n_rows=7, k=5, we have 7 rows of 6 elements, 15 diffs per row, 105 total
    # We can just recompute validity each time (it's fast enough)

    # Move generation:
    # Type 1: Change one entry in one row (keep order)
    # Type 2: Swap two rows
    # Type 3: Perturb a whole row (regenerate with small changes)

    step = 0
    while time.time() < deadline and t > t_min:
        step += 1
        if step % 1000 == 0 and time.time() > deadline:
            break

        # Choose a move
        move_type = rng.random()
        if move_type < 0.7:
            # Change one entry in one row
            ri = rng.randrange(n_rows)
            # Pick a position from 1 to n-1 (position 0 is always 0)
            pi = rng.randrange(1, n)
            row = cur_rows[ri]

            # Determine range for new value
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < n - 1 else max(row) + 50

            if hi <= lo:
                t *= cooling
                continue

            # Try a few random values in range
            delta = rng.randint(-max(1, hi - lo) // 2, max(1, hi - lo) // 2)
            new_val = row[pi] + delta
            new_val = max(lo, min(hi, new_val))

            if new_val == row[pi]:
                t *= cooling
                continue

            # Save old value
            old_val = row[pi]
            row[pi] = new_val

            # Check validity
            valid = True
            all_diffs = set()
            for r_idx, r in enumerate(cur_rows):
                rd = row_diffs_mask(r)
                if rd & all_diffs:
                    valid = False
                    break
                all_diffs |= rd
            if not valid:
                row[pi] = old_val
                t *= cooling
                continue

            new_cost = cost(cur_rows)
            delta_c = new_cost - cur_cost

            if delta_c < 0 or rng.random() < math.exp(-delta_c / t):
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in cur_rows]
            else:
                row[pi] = old_val

        elif move_type < 0.85:
            # Swap two rows
            if n_rows < 2:
                t *= cooling
                continue
            ri, rj = rng.sample(range(n_rows), 2)
            cur_rows[ri], cur_rows[rj] = cur_rows[rj], cur_rows[ri]
            # Swapping rows doesn't change cost (symmetric)
            # But it might help with future moves
            # Actually for this problem, rows are not ordered, so swap is neutral
            cur_rows[ri], cur_rows[rj] = cur_rows[rj], cur_rows[ri]
            t *= cooling
        else:
            # Perturb a whole row: try to rebuild it greedily with some randomness
            ri = rng.randrange(n_rows)
            old_row = list(cur_rows[ri])

            # Remove this row's diffs from consideration temporarily
            # Just regenerate the row with some randomization
            new_row = [0]
            used_other = set()
            for r_idx, r in enumerate(cur_rows):
                if r_idx == ri:
                    continue
                used_other |= row_diffs_mask(r)

            # Greedy with randomness
            x = 0
            for _ in range(n - 1):
                # Try several candidates
                best_candidate = None
                for _attempt in range(20):
                    candidate = x + rng.randint(1, 15)
                    ok = True
                    for a in new_row:
                        d = candidate - a
                        if d in used_other or d in row_diffs_mask(new_row):
                            ok = False
                            break
                    if ok:
                        best_candidate = candidate
                        break
                if best_candidate is None:
                    # Fallback: try sequential
                    found = False
                    for cand in range(x + 1, x + 100):
                        ok = True
                        for a in new_row:
                            d = cand - a
                            if d in used_other or d in row_diffs_mask(new_row):
                                ok = False
                                break
                        if ok:
                            best_candidate = cand
                            found = True
                            break
                    if not found:
                        break
                new_row.append(best_candidate)
                x = best_candidate

            if len(new_row) < n:
                # Couldn't complete, revert
                t *= cooling
                continue

            cur_rows[ri] = new_row
            new_cost = cost(cur_rows)
            delta_c = new_cost - cur_cost

            if delta_c < 0 or rng.random() < math.exp(-delta_c / t):
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(r) for r in cur_rows]
            else:
                cur_rows[ri] = old_row

        t *= cooling

    return best_rows


def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    # Always produce a valid greedy baseline immediately.
    greedy_rows = _greedy(n_rows, k)
    best = greedy_rows
    best_scope = _scope(best)

    # Try to improve via Simulated Annealing with a time budget.
    if time.time() < deadline - 5.0:
        try:
            improved = _sa_optimize(n_rows, k, greedy_rows, deadline, seed=42)
            if improved is not None and _validate(improved, k):
                if _scope(improved) < best_scope:
                    best = improved
                    best_scope = _scope(best)
        except Exception:
            pass

    # Second SA run with different seed if time permits
    if time.time() < deadline - 10.0:
        try:
            improved2 = _sa_optimize(n_rows, k, best, deadline - 5.0, seed=123)
            if improved2 is not None and _validate(improved2, k):
                if _scope(improved2) < best_scope:
                    best = improved2
                    best_scope = _scope(best)
        except Exception:
            pass

    # Final safety: if no improvement, return greedy.
    if not _validate(best, k):
        # regenerate a guaranteed-valid greedy
        best = _greedy(n_rows, k)
    return best
