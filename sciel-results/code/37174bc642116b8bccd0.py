"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    a_j - a_i (i < j) within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s out of 120s budget

    # Greedy baseline: each row starts at 0 and takes the smallest next mark
    # whose differences to earlier marks of the row are all still unused.
    def greedy_baseline():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, max(r[-1] for r in rows)

    baseline_rows, baseline_scope = greedy_baseline()
    best_rows = baseline_rows
    best_scope = baseline_scope

    # Try to improve via randomized restarts with backtracking
    # We'll use a randomized greedy with occasional restarts
    def randomized_greedy(seed_val):
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    # Occasionally, we can try to "reset" and pick a larger gap
                    # to leave more room for future rows
                    if rng.random() < 0.1 and len(row) < k:
                        # Skip a few positions to create more spacing
                        skip = rng.randint(1, 3)
                        for _s in range(skip):
                            x += 1
                            # Check if skipping this x is valid (it just means we don't use it)
                            pass
            rows.append(row)
        if len(rows) == n_rows and all(len(r) == k + 1 for r in rows):
            # Verify validity
            all_diffs = set()
            valid = True
            for r in rows:
                row_diffs = set()
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs or d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                if not valid:
                    break
                all_diffs |= row_diffs
            if valid:
                return rows, max(r[-1] for r in rows)
        return None, float('inf')

    # Try a few randomized restarts
    attempts = 0
    max_attempts = 200
    while time.time() < deadline and attempts < max_attempts:
        seed_val = random.randint(0, 10**9)
        rows, scope = randomized_greedy(seed_val)
        if rows is not None and scope < best_scope:
            best_rows = rows
            best_scope = scope
            max_attempts = min(max_attempts, 500)  # Allow more if improving
        attempts += 1

    return best_rows

# EVOLVE-BLOCK-END
