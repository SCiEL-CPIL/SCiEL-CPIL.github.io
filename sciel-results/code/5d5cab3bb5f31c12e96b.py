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

    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def try_build_row(used, rng, max_val, k_val, bias_low=False):
        """Try to build a row of k_val+1 entries (starting at 0) with max entry <= max_val,
        avoiding diffs in `used`. Returns row or None."""
        row = [0]
        for pos in range(k_val):
            lo = row[-1] + 1
            hi = max_val
            if lo > hi:
                return None
            found = False
            # Try random candidates, biased toward lower end
            if bias_low:
                for _ in range(40):
                    x = lo + int((rng.random() ** 1.5) * (hi - lo + 1))
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
            else:
                for _ in range(40):
                    x = rng.randint(lo, hi)
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
                while x <= hi:
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
                        found = True
                        break
            if not found:
                return None
        if row[-1] > max_val:
            return None
        return row

    def build_row_sequential(used, k_val):
        """Deterministic sequential build: smallest possible next value."""
        row = [0]
        for pos in range(k_val):
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
        return row

    def other_rows_used(rows, skip_idx):
        used = set()
        for i, r in enumerate(rows):
            if i == skip_idx:
                continue
            for d in row_diffs(r):
                used.add(d)
        return used

    def find_min_max_for_row(used, k_val, rng, low, high, bias_low=False):
        """Binary search the minimum max_val such that a valid row exists with max <= max_val."""
        # First check if high is feasible
        test_used = set(used)
        if try_build_row(test_used, rng, high, k_val, bias_low) is None:
            return None
        # Binary search
        lo = low
        hi = high
        best = high
        while lo <= hi and time.time() < deadline - 5:
            mid = (lo + hi) // 2
            test_used = set(used)
            if try_build_row(test_used, rng, mid, k_val, bias_low) is not None:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best

    # Initial construction: build rows one by one with sequential greedy
    rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = build_row_sequential(used, k)
        rows.append(row)
        for d in row_diffs(row):
            used.add(d)

    best_rows = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_rows)

    # Local search: repeatedly pick a row, regenerate it with a tighter bound
    # to reduce the overall scope.
    current_scope = best_scope
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 10 and no_improve < max_no_improve:
        # Pick a random row to try to improve
        idx = rng.randint(0, n_rows - 1)
        current_max = max(rows[idx])
        
        # The target: we want to reduce the overall scope.
        # Find the current overall scope excluding this row.
        other_max = 0
        for i in range(n_rows):
            if i != idx:
                other_max = max(other_max, max(rows[i]))
        
        # We can afford max_val = min(current_scope - 1, current_max - 1) if we want to improve
        # But at minimum, try to reduce this row's max.
        target_max = min(current_scope, current_max) - 1
        if target_max < m - 1:  # minimum possible is k (since entries are 0..k at least)
            target_max = m - 1
        
        used_others = other_rows_used(rows, idx)
        
        # Try to build a row with max <= target_max
        # Try multiple random attempts
        success = False
        new_row = None
        for attempt in range(20):
            test_used = set(used_others)
            bias_low = (attempt < 10)  # first 10 attempts bias low
            candidate = try_build_row(test_used, rng, target_max, k, bias_low)
            if candidate is not None:
                new_row = candidate
                success = True
                break
        
        if success and new_row is not None:
            new_scope = max(other_max, max(new_row))
            if new_scope < current_scope:
                rows[idx] = new_row
                current_scope = new_scope
                no_improve = 0
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            else:
                no_improve += 1
        else:
            no_improve += 1

    # Final compaction: for each row, try to reduce its max by 1, 2, 3... using binary search
    for idx in range(n_rows):
        if time.time() > deadline - 5:
            break
        current_row_max = max(rows[idx])
        used_others = other_rows_used(rows, idx)
        # Try to find a row with smaller max
        for target in range(current_row_max - 1, m - 2, -1):
            test_used = set(used_others)
            candidate = try_build_row(test_used, rng, target, k, True)
            if candidate is not None:
                rows[idx] = candidate
                break

    # Final validity check
    if is_valid(rows):
        final_scope = max(max(r) for r in rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = [list(r) for r in rows]

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
