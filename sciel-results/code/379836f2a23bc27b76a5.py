"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 118  # safety margin before 120s limit

    def greedy_row(used, temp=0.0, x_start=0):
        """Build one row greedily with optional temperature for exploration."""
        row = [0]
        x = x_start
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if temp > 0 and random.random() < temp and x < 200:
                    continue
                row.append(x)
                used |= new_diffs
        return row, used

    def diff_set(row):
        """Return the set of positive internal differences of a row."""
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def used_from_rows(rows):
        used = set()
        for r in rows:
            used |= diff_set(r)
        return used

    def validate(rows):
        used = set()
        for r in rows:
            for d in diff_set(r):
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Strategy 1: Pure greedy baseline ----------
    used = set()
    rows_greedy = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows_greedy.append(row)
    best_rows = [list(r) for r in rows_greedy]
    best_max = max_val(best_rows)

    # ---------- Strategy 2: Multi-temperature restarts ----------
    random.seed(42)
    attempts = 0
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    while time.time() < deadline - 15 and attempts < 200:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used = set()
        trial_rows = []
        success = True
        for _ in range(n_rows):
            if time.time() > deadline - 15:
                success = False
                break
            row, used = greedy_row(used, temp=temp)
            trial_rows.append(row)
        if not success:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]

    # ---------- Strategy 3: Row-swap local search ----------
    # For each row, try to find a replacement that fits the fixed others
    # and lowers the max. More candidates per row than a single greedy rebuild.
    def find_replacement(fixed_used, current_row, max_budget):
        """Try many candidate rows; return the best (lowest max) that fits."""
        best_local = None
        best_local_max = max(current_row) if current_row else float('inf')
        temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        for temp in temps:
            if time.time() > deadline - 10:
                break
            for _ in range(3):  # a few random restarts per temperature
                if time.time() > deadline - 10:
                    break
                trial_row, _ = greedy_row(fixed_used, temp=temp)
                if max(trial_row) >= max_budget:
                    continue
                if max(trial_row) < best_local_max:
                    best_local = trial_row
                    best_local_max = max(trial_row)
        return best_local

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_rows):
            if time.time() > deadline - 10:
                break
            other_rows = [r for j, r in enumerate(best_rows) if j != i]
            fixed_used = used_from_rows(other_rows)
            current_row = best_rows[i]
            # budget: must be strictly better than current max to count as improvement
            new_row = find_replacement(fixed_used, current_row, best_max)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if validate(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        improved = True

    # ---------- Strategy 4: Pair-swap refinement ----------
    # Rebuild two rows together (fixing the rest) to rebalance interactions.
    if time.time() < deadline - 15:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        random.shuffle(pairs)
        for (i, j) in pairs:
            if time.time() > deadline - 10:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = used_from_rows(other_rows)
            # Build two rows sequentially against fixed_used, trying a couple temps
            for temp in [0.0, 0.1, 0.2]:
                if time.time() > deadline - 10:
                    break
                used = set(fixed_used)
                r1, used = greedy_row(used, temp=temp)
                r2, used = greedy_row(used, temp=temp)
                new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                            for x, r in enumerate(best_rows)]
                if validate(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows

    # ---------- Strategy 5: Global random-restart escape ----------
    # Occasionally do a fresh randomized construction; keep the best overall.
    while time.time() < deadline - 5:
        temp = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        used = set()
        trial_rows = []
        success = True
        for _ in range(n_rows):
            if time.time() > deadline - 5:
                success = False
                break
            row, used = greedy_row(used, temp=temp)
            trial_rows.append(row)
        if not success:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]
        # After a fresh construction, do a quick row-swap polish
        for i in range(n_rows):
            if time.time() > deadline - 5:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i]
            fixed_used = used_from_rows(other_rows)
            new_row = find_replacement(fixed_used, best_rows[i], best_max)
            if new_row is not None:
                new_rows = [r if x != i else new_row for x, r in enumerate(best_rows)]
                if validate(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows

    return best_rows
# EVOLVE-BLOCK-END
