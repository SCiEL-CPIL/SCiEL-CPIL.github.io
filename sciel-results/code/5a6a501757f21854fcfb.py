import time
import sys
import random

def construct(n_rows, k):
    """
    Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.
    Uses a hybrid approach: fast greedy initialisation, then targeted local search with
    simulated-annealing-style moves to reduce the global maximum.
    """
    _START = time.time()
    _DEADLINE = 135.0  # safety margin before 150s kill

    def _elapsed():
        return time.time() - _START

    def _time_left():
        return _DEADLINE - _elapsed()

    def _row_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_diffs(rows):
        """Return set of all differences across all rows."""
        d = set()
        for r in rows:
            d |= _row_diffs(r)
        return d

    def _validate(rows, n_rows, k):
        """Check validity: rows strictly increasing from 0, all diffs distinct."""
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(k):
                if row[i] >= row[i + 1]:
                    return False
            d = _row_diffs(row)
            if len(d) != k * (k + 1) // 2:
                return False
            if d & seen:
                return False
            seen |= d
        return True

    def _scope(rows):
        return max(max(r) for r in rows)

    # --- Greedy construction ---
    def greedy_build(n_rows, k, rng, strategy="sequential"):
        """
        Build rows greedily.
        strategy: 'sequential' (build each row fully before next),
                  'interleaved_smallest' (extend row with smallest last element),
                  'interleaved_largest' (extend row with largest last element),
                  'interleaved_random' (extend a random row).
        """
        used = set()
        rows = [None] * n_rows

        if strategy == "sequential":
            for r in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            row.append(x)
                            for a in row[:-1]:
                                used.add(x - a)
                            break
                        x += 1
                rows[r] = row
            return rows
        else:
            rows = [[0] for _ in range(n_rows)]
            placed = n_rows
            total = n_rows * (k + 1)
            while placed < total:
                if _time_left() < 5.0:
                    return None
                candidates = [r for r in range(n_rows) if len(rows[r]) < k + 1]
                if not candidates:
                    break
                if strategy == "interleaved_smallest":
                    best_r = min(candidates, key=lambda r: rows[r][-1])
                elif strategy == "interleaved_largest":
                    best_r = max(candidates, key=lambda r: rows[r][-1])
                else:
                    best_r = rng.choice(candidates)
                row = rows[best_r]
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        placed += 1
                        break
                    x += 1
            return rows

    # --- Local search: reduce the global maximum ---
    def local_search(rows, n_rows, k, rng, time_budget=60.0):
        """
        Targeted descent: repeatedly try to reduce the last element of the row(s)
        containing the global maximum. Uses simulated-annealing-style acceptance
        to occasionally accept worse moves to escape local minima.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_max = _scope(rows)
        start = time.time()
        temp = 5.0  # initial "temperature"

        while time.time() - start < time_budget and _time_left() > 5.0:
            cur_max = _scope(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]
                temp = min(temp + 1.0, 10.0)  # reward improvement

            max_rows = [r for r in range(n_rows) if max(rows[r]) == cur_max]
            improved_this_round = False

            for r in max_rows:
                if _time_left() < 5.0:
                    break
                row = rows[r]
                if len(row) < 2:
                    continue
                old_val = row[-1]
                # Try reducing the last element
                for new_val in range(old_val - 1, row[-2], -1):
                    # Compute new diffs for this row
                    new_row = row[:-1] + [new_val]
                    new_row_d = _row_diffs(new_row)
                    # Diffs from other rows
                    other_d = set()
                    for r2 in range(n_rows):
                        if r2 != r:
                            other_d |= _row_diffs(rows[r2])
                    # Check: new_row_d must not intersect other_d
                    # And new_row_d must have exactly k*(k+1)/2 elements (no internal dupes)
                    if len(new_row_d) == k * (k + 1) // 2 and not (new_row_d & other_d):
                        # Accept
                        rows[r] = new_row
                        improved_this_round = True
                        break
                if improved_this_round:
                    break

            if not improved_this_round:
                # Try a more aggressive move: change an earlier element in the max row
                r = rng.choice(max_rows)
                row = rows[r]
                # Pick a random position to modify (not 0, not last if possible)
                pos = rng.randint(1, len(row) - 1)
                old_val = row[pos]
                # Try to decrease it
                for new_val in range(old_val - 1, (row[pos - 1] + 1 if pos > 0 else 0), -1):
                    # Ensure strict increase
                    if pos > 0 and new_val <= row[pos - 1]:
                        continue
                    if pos < len(row) - 1 and new_val >= row[pos + 1]:
                        continue
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if not (new_row[0] == 0 and all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1))):
                        continue
                    new_row_d = _row_diffs(new_row)
                    other_d = set()
                    for r2 in range(n_rows):
                        if r2 != r:
                            other_d |= _row_diffs(rows[r2])
                    if len(new_row_d) == k * (k + 1) // 2 and not (new_row_d & other_d):
                        rows[r] = new_row
                        improved_this_round = True
                        break

            if not improved_this_round:
                # Simulated annealing: accept a random valid perturbation
                # to escape local minimum
                r = rng.randrange(n_rows)
                row = rows[r]
                pos = rng.randint(1, len(row) - 1)
                old_val = row[pos]
                # Try a small random change
                delta = rng.randint(-5, 5)
                new_val = old_val + delta
                if new_val > 0 and (pos == 0 or new_val > row[pos - 1]) and (pos == len(row) - 1 or new_val < row[pos + 1]):
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if new_row[0] == 0 and all(new_row[i] < new_row[i + 1] for i in range(len(new_row) - 1)):
                        new_row_d = _row_diffs(new_row)
                        other_d = set()
                        for r2 in range(n_rows):
                            if r2 != r:
                                other_d |= _row_diffs(rows[r2])
                        if len(new_row_d) == k * (k + 1) // 2 and not (new_row_d & other_d):
                            # Accept with probability based on temperature
                            new_scope = _scope([rows[i] if i != r else new_row for i in range(n_rows)])
                            old_scope = _scope(rows)
                            if new_scope <= old_scope or (temp > 0 and rng.random() < (1.0 / (1.0 + (new_scope - old_scope) / temp))):
                                rows[r] = new_row

            # Cool down
            temp *= 0.95

        return best_rows

    # --- Main ---
    rng = random.Random(42)

    # Phase 1: Greedy initialisation with multiple strategies
    best_rows = None
    best_max = float('inf')

    strategies = ["sequential", "interleaved_smallest", "interleaved_largest", "interleaved_random"]
    for strat in strategies:
        if _time_left() < 10.0:
            break
        rows = greedy_build(n_rows, k, rng, strategy=strat)
        if rows is not None and _validate(rows, n_rows, k):
            s = _scope(rows)
            if s < best_max:
                best_max = s
                best_rows = [list(r) for r in rows]

    # Also try a few random row-order sequential builds
    for _ in range(10):
        if _time_left() < 10.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        used = set()
        rows = [None] * n_rows
        for r in order:
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        break
                    x += 1
            rows[r] = row
        if _validate(rows, n_rows, k):
            s = _scope(rows)
            if s < best_max:
                best_max = s
                best_rows = [list(r) for r in rows]

    if best_rows is None:
        # Fallback: pure sequential greedy
        used = set()
        best_rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        break
                    x += 1
            best_rows.append(row)
        best_max = _scope(best_rows)

    # Phase 2: Local search to reduce the maximum
    if _time_left() > 15.0:
        improved = local_search(best_rows, n_rows, k, rng, time_budget=_time_left() - 15.0)
        if improved is not None and _validate(improved, n_rows, k):
            s = _scope(improved)
            if s < best_max:
                best_max = s
                best_rows = improved

    return best_rows
