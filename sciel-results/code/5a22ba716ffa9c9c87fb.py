"""Difference triangle set (4,7): minimize largest entry with distinct differences."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Greedy construction for a single row given the set of already-used differences.
    def greedy_row(used):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        return row

    def all_diffs(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def is_valid(rows):
        d = all_diffs(rows)
        return len(d) == n_rows * (k + 1) * k // 2

    def scope(rows):
        return max(max(r) for r in rows)

    # Build initial rows greedily (fast, valid).
    def build_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            rows.append(greedy_row(used))
        return rows

    # Annealing with element-level moves.
    def anneal(rows, start_temp, steps, rng):
        rows = [list(r) for r in rows]
        cur_score = scope(rows)
        best = [list(r) for r in rows]
        best_score = cur_score
        total_diffs = n_rows * (k + 1) * k // 2
        used = all_diffs(rows)

        for step in range(steps):
            if time.time() > deadline:
                break
            # Pick a random row and a random element (not the first, which is fixed at 0).
            ri = rng.randrange(n_rows)
            ei = rng.randrange(1, k + 1)
            row = rows[ri]
            old = row[ei]

            # Compute the set of differences involving this element.
            old_diffs = {abs(old - a) for a in row if a != old}

            # Candidate new value: try to reduce scope by pulling inward, or jitter.
            if rng.random() < 0.5 and cur_score > 0:
                # Try to move toward smaller values to reduce max.
                hi = old - 1
                lo = 0
                if hi > lo:
                    new = rng.randint(lo, hi)
                else:
                    new = old
            else:
                # Random jitter within a window.
                span = max(cur_score, 20)
                new = old + rng.randint(-span // 2, span // 2)
                if new < 0:
                    new = old
            if new == old:
                continue

            # Check ordering: row must remain strictly increasing.
            prev = row[ei - 1]
            nxt = row[ei + 1] if ei + 1 <= k else None
            if new <= prev:
                continue
            if nxt is not None and new >= nxt:
                continue

            # Tentatively apply.
            row[ei] = new
            new_diffs = {abs(new - a) for a in row if a != new}

            # Differences that become newly used: new_diffs not in old_diffs.
            added = new_diffs - old_diffs
            # Check for collisions: any added diff already used elsewhere.
            conflict = any(d in used and d not in old_diffs for d in added)
            # Also ensure added diffs are not among other used diffs.
            if not conflict:
                # Compute score change.
                new_score = max(max(r) for r in rows)
                delta = new_score - cur_score
                if delta <= 0 or rng.random() < math.exp(-delta / max(start_temp, 1e-9)):
                    # Accept: commit.
                    used -= old_diffs
                    used |= new_diffs
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best = [list(r) for r in rows]
                else:
                    # Reject: revert.
                    row[ei] = old
            else:
                # Revert.
                row[ei] = old

            # Cool down.
            if step % 100 == 0 and start_temp > 0.01:
                start_temp *= 0.99

        return best, best_score

    rng = random.Random(12345)

    # First, get a valid baseline.
    rows = build_initial()
    best = [list(r) for r in rows]
    best_score = scope(best)

    # Annealing phases with restarts.
    phase_temps = [50.0, 30.0, 15.0, 8.0, 4.0]
    for temp in phase_temps:
        if time.time() > deadline:
            break
        # Use current best as starting point.
        start = [list(r) for r in best]
        cand, cand_score = anneal(start, temp, 200000, rng)
        if cand_score < best_score:
            best = cand
            best_score = cand_score

    # Final restarts from fresh greedy to explore different basins.
    restarts = 3
    for _ in range(restarts):
        if time.time() > deadline:
            break
        fresh = build_initial()
        cand, cand_score = anneal(fresh, 20.0, 100000, rng)
        if cand_score < best_score:
            best = cand
            best_score = cand_score

    # Verify validity of best; if somehow invalid, fall back to greedy.
    if not is_valid(best):
        best = build_initial()

    return best
# EVOLVE-BLOCK-END
