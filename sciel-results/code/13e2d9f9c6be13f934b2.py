import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # --- Fast greedy to get an initial valid solution (upper bound) ---
    def greedy_solve(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                # Try random candidates in increasing preference for small values
                for attempt in range(200):
                    # Bias towards smaller values
                    hi = row[-1] + 1 + int(200 * (rng.random() ** 1.5))
                    if hi < row[-1] + 1:
                        hi = row[-1] + 1
                    x = rng.randint(row[-1] + 1, hi)
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
                if not found:
                    # Sequential fallback
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            break
            rows.append(row)
        return rows

    # --- Backtracking solver with scope bound M ---
    def backtrack_solve(M, seed, time_budget):
        rng = random.Random(seed)
        start = time.time()
        used = set()
        rows = []

        def solve_row(pos_idx, current_row):
            """Backtrack to complete a row given current_row (starts with 0)."""
            if time.time() - start > time_budget:
                return False
            if len(current_row) == m:
                return True

            last = current_row[-1]
            # Candidate values: from last+1 to M
            # Use MRV-like: try values that create the fewest conflicts first
            # For efficiency, iterate candidates in a randomized but bounded order
            candidates = list(range(last + 1, M + 1))
            # Shuffle with bias towards smaller values
            # Weight: smaller values get higher probability
            if len(candidates) > 1:
                weights = [1.0 / (1 + i * 0.1) for i in range(len(candidates))]
                # Partial shuffle: pick a few random ones
                for _ in range(min(5, len(candidates))):
                    i = rng.randrange(len(candidates))
                    j = rng.randrange(len(candidates))
                    candidates[i], candidates[j] = candidates[j], candidates[i]

            for x in candidates:
                # Check if x - a is in used for all a in current_row
                ok = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue

                # Tentatively add
                for d in new_diffs:
                    used.add(d)
                current_row.append(x)

                if solve_row(pos_idx + 1, current_row):
                    return True

                # Undo
                current_row.pop()
                for d in new_diffs:
                    used.discard(d)

            return False

        def solve_all_rows(row_idx):
            if time.time() - start > time_budget:
                return False
            if row_idx == n_rows:
                return True

            # Solve this row
            row = [0]
            if solve_row(0, row):
                rows.append(row)
                if solve_all_rows(row_idx + 1):
                    return True
                rows.pop()

            return False

        if solve_all_rows(0):
            return rows
        return None

    # --- Main strategy: binary search on scope with randomized backtracking ---
    # First, get a greedy solution for an initial upper bound
    best_rows = greedy_solve(42)
    best_scope = max(max(r) for r in best_rows)

    # Counting bound
    counting_bound = math.ceil(math.sqrt(2 * total_diffs))
    if counting_bound < best_scope:
        best_scope_candidate = counting_bound
    else:
        best_scope_candidate = best_scope

    # Binary search for the minimal feasible scope
    # We'll try scopes from max(counting_bound, some_min) up to best_scope
    lo = max(counting_bound, m - 1)  # minimum possible scope
    hi = best_scope

    # Try a few random seeds for each scope in the binary search
    found_scope = hi
    found_rows = best_rows

    # Do a binary search with time-limited backtracking
    while lo <= hi and time.time() < deadline - 20:
        mid = (lo + hi) // 2
        # Try to solve with scope = mid
        success = False
        for seed in range(5):  # Try up to 5 random restarts
            time_left = deadline - time.time() - 20
            if time_left < 2:
                break
            time_budget = min(10.0, time_left / (5 - seed))
            rows = backtrack_solve(mid, seed, time_budget)
            if rows is not None and is_valid(rows):
                success = True
                found_scope = mid
                found_rows = rows
                break

        if success:
            hi = mid - 1
        else:
            lo = mid + 1

    # Update best if we found something better
    if found_scope < best_scope:
        best_scope = found_scope
        best_rows = found_rows

    # If we have time, try to improve further with more random seeds around the found scope
    if time.time() < deadline - 10:
        for trial_scope in range(found_scope, min(found_scope + 5, hi + 1)):
            if time.time() > deadline - 10:
                break
            for seed in range(20):
                time_left = deadline - time.time() - 5
                if time_left < 1:
                    break
                time_budget = min(5.0, time_left / 10)
                rows = backtrack_solve(trial_scope, seed + 1000, time_budget)
                if rows is not None and is_valid(rows):
                    scope = max(max(r) for r in rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows
                        break

    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_solve(42)

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
