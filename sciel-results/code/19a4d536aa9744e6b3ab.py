"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit

    def greedy_row(used, temp=0.0):
        """Build one row greedily with optional temperature for exploration."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # With temperature, sometimes skip valid x to explore
                if temp > 0 and random.random() < temp and x < 150:
                    continue
                row.append(x)
                used |= new_diffs
        return row, used

    def validate(rows):
        """Check if all positive differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # Strategy 1: Pure greedy baseline
    used = set()
    rows_greedy = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows_greedy.append(row)
    best_rows = list(rows_greedy)
    best_max = max_val(rows_greedy)

    # Strategy 2: Greedy with varying temperatures
    random.seed(42)
    attempts = 0
    temps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    while time.time() < deadline - 5 and attempts < 100:
        attempts += 1
        temp = temps[attempts % len(temps)]
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
            best_rows = list(trial_rows)

    # Strategy 3: Local search - rebuild individual rows
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n_rows):
                if time.time() > deadline - 10:
                    break
                # Compute used set from all rows except row i
                other_rows = [r for j, r in enumerate(best_rows) if j != i]
                used = set()
                for r in other_rows:
                    for a in r:
                        for b in r:
                            if a < b:
                                used.add(b - a)

                # Try to rebuild row i with different temperatures
                for temp in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
                    if time.time() > deadline - 10:
                        break
                    trial_row, _ = greedy_row(used, temp=temp)
                    new_rows = [r if j != i else trial_row for j, r in enumerate(best_rows)]
                    if validate(new_rows):
                        mv = max_val(new_rows)
                        if mv < best_max:
                            best_max = mv
                            best_rows = new_rows
                            improved = True

    return best_rows
# EVOLVE-BLOCK-END
