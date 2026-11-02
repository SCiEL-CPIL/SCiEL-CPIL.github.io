"""Fresh design: Simulated Annealing with randomized restarts and a deadline guard."""
import time
import random

DEADLINE_FRACTION = 0.92
RANDOM_SEED = 12345


def construct(n_rows, k):
    random.seed(RANDOM_SEED)
    start_time = time.time()
    deadline = start_time + 120.0 * DEADLINE_FRACTION

    # Small instance: use deterministic greedy (matches seed behavior exactly).
    if n_rows <= 2 and k <= 2:
        return _greedy(n_rows, k)

    n_marks = k + 1
    n_diffs = k * (k + 1) // 2  # positive differences per row
    # Total differences that must be distinct across all rows.
    total_diffs = n_rows * n_diffs

    # Upper bound for the scope: greedy gives a valid (possibly suboptimal) scope.
    greedy_rows = _greedy(n_rows, k)
    initial_scope = max(max(r) for r in greedy_rows)
    # We search up to this bound; SA will try to find a smaller valid scope.
    upper_bound = initial_scope

    # State: rows of length n_marks, each strictly increasing starting at 0.
    # We'll initialize each row with a random valid increasing sequence in [0, upper_bound].
    def random_row(bound):
        # Pick n_marks - 1 distinct values from 1..bound, sort them, prepend 0.
        if bound < n_marks - 1:
            # Fallback: minimal row
            return list(range(n_marks))
        vals = random.sample(range(1, bound + 1), n_marks - 1)
        vals.sort()
        return [0] + vals

    # Compute conflicts (number of duplicate differences) and scope.
    def evaluate(rows):
        scope = 0
        seen = set()
        conflicts = 0
        for row in rows:
            m = row[-1]
            if m > scope:
                scope = m
            for i in range(n_marks):
                a = row[i]
                for j in range(i + 1, n_marks):
                    d = row[j] - a
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
        return scope, conflicts

    # Initialize rows.
    rows = [random_row(upper_bound) for _ in range(n_rows)]
    best_rows = None
    best_scope = None
    best_conflicts = None

    # Evaluate initial.
    scope, conflicts = evaluate(rows)
    if conflicts == 0:
        best_rows = [list(r) for r in rows]
        best_scope = scope
        best_conflicts = 0

    # Annealing parameters.
    T = float(upper_bound)
    T_min = 1e-3
    cooling = 0.999
    # Step budget: we rely on the deadline, but also cap iterations to avoid
    # pathological loops.
    max_iters = 5_000_000
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 20000

    def energy(rows):
        s, c = evaluate(rows)
        return s + 10000.0 * c

    current_energy = energy(rows)

    while iter_count < max_iters:
        # Deadline guard.
        if time.time() > deadline:
            break

        # Choose a move.
        move_type = random.random()
        r_idx = random.randrange(n_rows)
        row = rows[r_idx]

        if move_type < 0.5:
            # Move a single non-zero mark to a random position in [1, upper_bound].
            pos = random.randrange(1, n_marks)
            old_val = row[pos]
            new_val = random.randint(1, upper_bound)
            # Ensure strict increase: check neighbors.
            if pos > 0 and new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < n_marks - 1 and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val == old_val or new_val < 1:
                iter_count += 1
                continue
            row[pos] = new_val
            new_energy = energy(rows)
            delta = new_energy - current_energy
            if delta <= 0 or random.random() < _boltzmann(delta, T):
                current_energy = new_energy
                # Track best.
                s, c = evaluate(rows)
                if c == 0:
                    if best_scope is None or s < best_scope:
                        best_scope = s
                        best_conflicts = 0
                        best_rows = [list(r) for r in rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                row[pos] = old_val
                no_improve_count += 1
        elif move_type < 0.8:
            # Swap two non-zero marks in the row.
            if n_marks < 3:
                iter_count += 1
                continue
            p, q = random.sample(range(1, n_marks), 2)
            row[p], row[q] = row[q], row[p]
            new_energy = energy(rows)
            delta = new_energy - current_energy
            if delta <= 0 or random.random() < _boltzmann(delta, T):
                current_energy = new_energy
                s, c = evaluate(rows)
                if c == 0:
                    if best_scope is None or s < best_scope:
                        best_scope = s
                        best_conflicts = 0
                        best_rows = [list(r) for r in rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                row[p], row[q] = row[q], row[p]
                no_improve_count += 1
        else:
            # Randomize the entire row.
            old_row = list(row)
            new_row = random_row(upper_bound)
            rows[r_idx] = new_row
            new_energy = energy(rows)
            delta = new_energy - current_energy
            if delta <= 0 or random.random() < _boltzmann(delta, T):
                current_energy = new_energy
                s, c = evaluate(rows)
                if c == 0:
                    if best_scope is None or s < best_scope:
                        best_scope = s
                        best_conflicts = 0
                        best_rows = [list(r) for r in rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                rows[r_idx] = old_row
                no_improve_count += 1

        # Cooling.
        T = max(T_min, T * cooling)
        iter_count += 1

        # Restart if stuck.
        if no_improve_count >= max_no_improve:
            # Randomize all rows to escape the local minimum.
            for i in range(n_rows):
                rows[i] = random_row(upper_bound)
            current_energy = energy(rows)
            no_improve_count = 0
            T = max(T, float(upper_bound) * 0.5)

    # Final: return the best valid solution found; if none, fall back to greedy.
    if best_rows is not None:
        return best_rows
    return greedy_rows


def _boltzmann(delta, T):
    if T <= 0:
        return 0.0
    # exp(-delta / T), clipped to avoid overflow.
    import math
    x = -delta / T
    if x > 0:
        return 1.0
    if x < -700:
        return 0.0
    return math.exp(x)


def _greedy(n_rows, k):
    """Deterministic greedy construction (matches the seed program)."""
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
    return rows
