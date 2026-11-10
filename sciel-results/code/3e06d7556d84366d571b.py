"""Difference triangle set (7,5) – local search / simulated-annealing style compressor."""
import random
import time

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150 s kill

    # ---- Build an initial valid solution with the greedy method (fast) ----
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows, used

    rows, used = greedy()
    best = [r[:] for r in rows]
    best_max = max(max(r) for r in best)

    # ---- Helpers ----
    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def rebuild_used(rows):
        u = set()
        for r in rows:
            u |= row_diffs(r)
        return u

    def scope(rows):
        return max(max(r) for r in rows)

    # ---- Local search: try to lower a single mark in a row ----
    def try_lower_mark(rows, used, r_idx, pos):
        """Try to replace rows[r_idx][pos] with a smaller value that keeps validity.
        Returns (new_val, old_val) or None."""
        row = rows[r_idx]
        old = row[pos]
        # The mark must stay > previous and < next (if any), and >=1
        lo = row[pos - 1] + 1 if pos > 0 else 1
        hi = row[pos + 1] - 1 if pos + 1 < len(row) else old
        if hi < lo:
            return None
        # Differences that involve this mark
        old_d = set()
        for i in range(len(row)):
            if i != pos:
                old_d.add(abs(row[pos] - row[i]))
        # Try values from lo..hi-1 (strictly smaller than old)
        for v in range(lo, min(hi + 1, old)):
            new_d = set()
            for i in range(len(row)):
                if i != pos:
                    new_d.add(abs(v - row[i]))
            # New diffs must not collide with (used - old_d)
            if not (new_d & (used - old_d)):
                return v, old
        return None

    rng = random.Random(12345)
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random row and a random non-first position
        r_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)  # positions 1..k (0 is fixed at 0)
        res = try_lower_mark(rows, used, r_idx, pos)
        if res is None:
            continue
        v, old = res
        row = rows[r_idx]
        # Apply move
        old_d = set()
        for i in range(len(row)):
            if i != pos:
                old_d.add(abs(old - row[i]))
        new_d = set()
        for i in range(len(row)):
            if i != pos:
                new_d.add(abs(v - row[i]))
        used -= old_d
        used |= new_d
        row[pos] = v
        cur_max = scope(rows)
        if cur_max < best_max:
            best = [r[:] for r in rows]
            best_max = cur_max
            # Occasionally reset rows to best to avoid drift? Keep current for momentum.
        # With some probability, revert to best to re-explore
        if rng.random() < 0.05:
            rows = [r[:] for r in best]
            used = rebuild_used(rows)

    # Return best found
    return best

# EVOLVE-BLOCK-END
