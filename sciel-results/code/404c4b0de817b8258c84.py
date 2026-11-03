"""Rewritten solver for the difference-triangle-set (n_rows, k) task.

Strategy:
  1. Fast greedy construction to get a valid initial solution.
  2. Iterated Local Search (ILS) with:
       - Move operator: pick a random row and a random position, try to relocate that
         element to a nearby (or globally random) new value, accepting if the total
         "cost" (sum of row spans) decreases or via simulated-annealing probability.
       - Restart: if no improvement for many iterations, perturb the best solution by
         randomly shuffling a few elements and re-optimizing.
  3. Deadline guard: return the best valid solution found before the time limit.

The objective is to minimize the maximum entry (scope). We use a soft surrogate:
  cost = sum over rows of (row[-1] - row[0])  ≈  scope when rows are balanced,
but more directly we minimize max(row[-1]) subject to validity. We track the best
scope explicitly.
"""

import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5-second safety margin

    # --- Helper: check if a set of rows is valid ---
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # --- Greedy construction ---
    def greedy(n_rows, k, rng=None):
        if rng is None:
            rng = random.Random(42)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # --- ILS with element moves ---
    def ily(rows, n_rows, k, rng, time_limit):
        """Iterated Local Search to minimize scope."""
        best_rows = [list(r) for r in rows]
        best_scope = scope(best_rows)
        current_rows = [list(r) for r in rows]
        current_scope = best_scope

        # Precompute: for each row, the set of differences
        def row_diffs(row):
            return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

        def all_diffs(rows):
            s = set()
            for row in rows:
                s |= row_diffs(row)
            return s

        cur_diffs = all_diffs(current_rows)

        # Annealing schedule
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_limit:
            # Pick a random row and position
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)  # not position 0 (which is fixed at 0)

            old_val = current_rows[ri][pi]

            # Try to find a new value
            # First, try nearby values
            found = False
            for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 20, -20]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Check increasing: new_val must be > current_rows[ri][pi-1] and < current_rows[ri][pi+1] if pi < k
                if pi > 0 and new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue

                # Check differences
                row = current_rows[ri]
                old_row_diffs = row_diffs(row)
                # Remove old diffs involving position pi
                removed = set()
                for j in range(len(row)):
                    if j != pi:
                        d = abs(row[j] - old_val)
                        if d > 0:
                            removed.add(d)

                # Add new diffs
                added = set()
                for j in range(len(row)):
                    if j != pi:
                        d = abs(row[j] - new_val)
                        if d > 0:
                            added.add(d)

                # Check if any added diff is in (cur_diffs - removed) or in added itself (duplicate)
                remaining = cur_diffs - removed
                if added & remaining:
                    continue
                if len(added) != len(added):  # always true, but check for internal duplicates
                    continue
                # Also check if added has duplicates within itself
                if len(added) < sum(1 for _ in added):  # this is always true for a set
                    continue

                # This move is valid
                # Compute new scope
                new_row = list(row)
                new_row[pi] = new_val
                new_scope = max(max(r) for r in current_rows) if ri != 0 else max(max(current_rows[ri]) if i == ri else max(r) for i, r in enumerate(current_rows))
                # Simpler:
                new_scope = 0
                for idx, r in enumerate(current_rows):
                    if idx == ri:
                        new_scope = max(new_scope, new_row[-1])
                    else:
                        new_scope = max(new_scope, r[-1])

                # Accept?
                if new_scope < current_scope:
                    # Accept
                    cur_diffs = (cur_diffs - removed) | added
                    current_rows[ri] = new_row
                    current_scope = new_scope
                    found = True
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    break
                else:
                    # Simulated annealing acceptance
                    delta = new_scope - current_scope
                    if rng.random() < math.exp(-delta / T):
                        cur_diffs = (cur_diffs - removed) | added
                        current_rows[ri] = new_row
                        current_scope = new_scope
                        found = True
                        no_improve += 1
                        break

            if not found:
                no_improve += 1

            # Cool down
            T = max(T_min, T * cooling)

            # Restart if stuck
            if no_improve > max_no_improve:
                # Perturb best solution
                current_rows = [list(r) for r in best_rows]
                current_scope = best_scope
                cur_diffs = all_diffs(current_rows)
                # Randomly perturb a few elements
                for _ in range(3):
                    ri = rng.randrange(n_rows)
                    pi = rng.randrange(1, k + 1)
                    old_val = current_rows[ri][pi]
                    # Try to find a valid new value
                    for attempt in range(50):
                        new_val = old_val + rng.randint(-20, 20)
                        if new_val <= 0:
                            continue
                        if pi > 0 and new_val <= current_rows[ri][pi - 1]:
                            continue
                        if pi < k and new_val >= current_rows[ri][pi + 1]:
                            continue
                        row = current_rows[ri]
                        old_row_diffs = row_diffs(row)
                        removed = set()
                        for j in range(len(row)):
                            if j != pi:
                                d = abs(row[j] - old_val)
                                if d > 0:
                                    removed.add(d)
                        added = set()
                        for j in range(len(row)):
                            if j != pi:
                                d = abs(row[j] - new_val)
                                if d > 0:
                                    added.add(d)
                        remaining = cur_diffs - removed
                        if not (added & remaining):
                            cur_diffs = (cur_diffs - removed) | added
                            current_rows[ri][pi] = new_val
                            break
                no_improve = 0
                T = 50.0

        return best_rows, best_scope

    # Main
    rng = random.Random(12345)
    rows = greedy(n_rows, k, rng)
    best_rows, best_scope = ily(rows, n_rows, k, rng, deadline)

    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        rows = greedy(n_rows, k, rng)
        if is_valid(rows):
            best_rows = rows
        else:
            # Should not happen, but return something
            return [[0] + list(range(1, k + 1)) for _ in range(n_rows)]

    return best_rows
