import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Hard deadline with safety margin. Evaluator kills at 150s.
    deadline = start_time + 135.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Check if all differences are distinct."""
        seen = set()
        for r in rows:
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_init(rand_prob=0.15, window=30):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        random.seed(int(time.time() * 1000) % (2**31))
        used_diffs = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]

                candidates = []
                x = last + 1
                limit = 300
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1

                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break

                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]

                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)

            rows.append(row)

        return rows

    def polish(rows):
        """
        Exhaustive single-element descent: try to reduce each element (except first)
        to the smallest valid value. Repeat until no improvement.
        """
        improved = True
        while improved:
            improved = False
            for r_idx in range(n_rows):
                for p_idx in range(1, row_len):
                    old_val = rows[r_idx][p_idx]
                    # Try to reduce this element
                    for new_val in range(rows[r_idx][p_idx - 1] + 1, old_val):
                        # Check if new_val is valid
                        valid = True
                        # Check against other elements in the same row
                        for i in range(p_idx):
                            d = new_val - rows[r_idx][i]
                            # Check if d conflicts with any used diff
                            # We need to check against all diffs from all rows
                            # except those involving this specific element
                            # This is complex, so let's just rebuild the diff set
                            pass

                        # Simpler approach: temporarily replace and check
                        rows[r_idx][p_idx] = new_val
                        if is_valid(rows):
                            improved = True
                            break
                        else:
                            rows[r_idx][p_idx] = old_val

    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        Returns rows if found, None otherwise.
        """
        random.seed(seed)
        start = time.time()

        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()

        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False

            if row_idx == n_rows:
                return True

            if pos == row_len:
                return solve(row_idx + 1, 1)

            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope

            if lower > upper:
                return False

            candidates = []
            scan_limit = min(upper, lower + 40)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 8:
                        break

            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 8:
                            break

            if not candidates:
                return False

            random.shuffle(candidates)

            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)

                rows[row_idx][pos] = val

                if solve(row_idx, pos + 1):
                    return True

                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)

            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 1: Get a baseline
    best_rows = greedy_init(0.15, 30)
    best_scope = get_scope(best_rows)

    # Phase 2: Polish the baseline
    polish(best_rows)
    best_scope = get_scope(best_rows)

    # Phase 3: Iterative Improvement using Backtracking
    current_target = best_scope - 1
    seed_counter = 0

    while time.time() < deadline - 5:
        if current_target < 112:
            break

        search_time = 5.0
        if time.time() > deadline - 10:
            search_time = deadline - time.time() - 2
            if search_time <= 0:
                break

        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1

        if result:
            # Polish the new result
            polish(result)
            new_scope = get_scope(result)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = result
                current_target = best_scope - 1
            else:
                break
        else:
            # Retry once with a different seed
            if time.time() < deadline - 10:
                result_retry = backtracking_search(n_rows, k, current_target, 5.0, seed=seed_counter)
                seed_counter += 1
                if result_retry:
                    polish(result_retry)
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    break
            else:
                break

    return best_rows


if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
