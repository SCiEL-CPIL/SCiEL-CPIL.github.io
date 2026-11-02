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
    m = k + 1
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

    # Global backtracking with MRV:
    # State: list of rows (each is list of ints, may be partial), used set
    # We assign one entry at a time. Variable = (row_idx, pos_idx) where pos_idx is next to fill.
    # Value = integer x.
    # MRV: choose the variable with the fewest available values (smallest domain) given current used set.
    # For efficiency, we only consider variables that are "next" in their row (pos = len(row)).

    def backtrack_global(M, seed, time_budget):
        rng = random.Random(seed)
        start = time.time()
        used = set()
        rows = [[0] for _ in range(n_rows)]  # each row starts with [0]
        # pos_idx for each row is len(row)-1 (0-indexed position of last filled)
        # We need to fill rows[ri] up to length m.

        def get_available(ri, last_val):
            """Return list of candidate values for next entry in row ri, given last_val."""
            # Candidates: last_val+1 .. M
            # Filter by: for all a in rows[ri], x - a not in used
            row = rows[ri]
            cands = []
            for x in range(last_val + 1, M + 1):
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    cands.append(x)
            return cands

        def select_variable():
            """MRV: pick (ri) with fewest candidates. Tie-break: smallest last_val, then random."""
            best_ri = -1
            best_count = float('inf')
            best_last = float('inf')
            for ri in range(n_rows):
                if len(rows[ri]) >= m:
                    continue
                last_val = rows[ri][-1]
                cands = get_available(ri, last_val)
                cnt = len(cands)
                if cnt == 0:
                    return ri, []  # dead end
                if cnt < best_count or (cnt == best_count and last_val < best_last):
                    best_count = cnt
                    best_last = last_val
                    best_ri = ri
            if best_ri == -1:
                return -1, []  # all rows complete
            cands = get_available(best_ri, rows[best_ri][-1])
            return best_ri, cands

        def solve():
            if time.time() - start > time_budget:
                return False
            # Check if all rows complete
            for ri in range(n_rows):
                if len(rows[ri]) < m:
                    break
            else:
                return True

            ri, cands = select_variable()
            if ri == -1:
                return True
            if not cands:
                return False

            # Randomize candidate order with bias towards smaller values
            # Use weighted shuffle: smaller x gets higher weight
            if len(cands) > 1:
                # Partial randomization: swap a few random pairs
                for _ in range(min(3, len(cands))):
                    i = rng.randrange(len(cands))
                    j = rng.randrange(len(cands))
                    cands[i], cands[j] = cands[j], cands[i]

            for x in cands:
                # Check and add
                new_diffs = []
                row = rows[ri]
                for a in row:
                    d = x - a
                    if d in used:
                        break
                    new_diffs.append(d)
                else:
                    # All good
                    for d in new_diffs:
                        used.add(d)
                    row.append(x)
                    if solve():
                        return True
                    row.pop()
                    for d in new_diffs:
                        used.discard(d)
            return False

        if solve():
            return rows
        return None

    # Fast greedy for initial upper bound
    def greedy_solve(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                for attempt in range(300):
                    hi = row[-1] + 1 + int(150 * (rng.random() ** 1.5))
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

    # Main: binary search on scope with randomized global backtracking
    best_rows = greedy_solve(42)
    best_scope = max(max(r) for r in best_rows)

    counting_bound = math.ceil(math.sqrt(2 * total_diffs))
    lo = max(counting_bound, m - 1)
    hi = best_scope

    found_scope = hi
    found_rows = best_rows

    while lo <= hi and time.time() < deadline - 25:
        mid = (lo + hi) // 2
        success = False
        for seed in range(8):
            time_left = deadline - time.time() - 25
            if time_left < 2:
                break
            time_budget = min(12.0, time_left / max(1, 8 - seed))
            rows = backtrack_global(mid, seed, time_budget)
            if rows is not None and is_valid(rows):
                success = True
                found_scope = mid
                found_rows = rows
                break
        if success:
            hi = mid - 1
        else:
            lo = mid + 1

    if found_scope < best_scope:
        best_scope = found_scope
        best_rows = found_rows

    # If time remains, try to improve around found_scope
    if time.time() < deadline - 10:
        for trial_scope in range(found_scope, min(found_scope + 3, hi + 1)):
            if time.time() > deadline - 10:
                break
            for seed in range(15):
                time_left = deadline - time.time() - 5
                if time_left < 1:
                    break
                time_budget = min(6.0, time_left / 10)
                rows = backtrack_global(trial_scope, seed + 1000, time_budget)
                if rows is not None and is_valid(rows):
                    scope = max(max(r) for r in rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = rows
                        break

    if not is_valid(best_rows):
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
