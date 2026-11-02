"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, with all pairwise differences distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Greedy initialization
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def compute_all_used(rows):
        used = set()
        for r in rows:
            used |= row_diffs_set(r)
        return used

    def global_max(rows):
        return max(max(r) for r in rows)

    def max_except(rows, exclude_idx):
        m = 0
        for i, r in enumerate(rows):
            if i != exclude_idx:
                if r[-1] > m:
                    m = r[-1]
        return m

    def validate(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            diffs = []
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
                    diffs.append(d)
            if len(diffs) != len(set(diffs)):
                return False
        return True

    def random_valid_row(forbidden, max_limit, rng, max_candidates=10):
        """Find a random valid row with max entry <= max_limit."""
        if max_limit < k:
            return None
        row = [0]
        used_in_row = set()

        for _ in range(k):
            candidates = []
            start = row[-1] + 1
            end = max_limit + 1

            # Scan for valid candidates
            for x in range(start, end):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & used_in_row:
                    continue
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break

            if not candidates:
                return None

            chosen_x = rng.choice(candidates)
            row.append(chosen_x)
            used_in_row |= {chosen_x - a for a in row[:-1]}

        return row

    # Start with greedy
    rows = greedy_init()
    best = [r[:] for r in rows]
    best_val = global_max(best)
    best_used = compute_all_used(best)

    cur = [r[:] for r in best]
    cur_val = best_val
    cur_row_diffs = [row_diffs_set(r) for r in cur]
    cur_used = set(best_used)

    rng = random.Random(42)

    # Main search loop: iterative local search with restarts
    T = 10.0
    T_min = 0.1
    cool = 0.999

    no_improve = 0
    max_no_improve = 200000

    while time.time() < deadline:
        # Pick a random row to replace
        ri = rng.randrange(n_rows)
        forbidden = cur_used - cur_row_diffs[ri]

        # Try to find a new row
        # First try with current max (or slightly less for improvement)
        target_max = cur_val
        if rng.random() < 0.3:
            target_max = cur_val - 1  # Try to improve

        new_row = random_valid_row(forbidden, target_max, rng, max_candidates=12)

        if new_row is None:
            # Try with slightly larger limit
            new_row = random_valid_row(forbidden, cur_val + 3, rng, max_candidates=8)

        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        other_max = max_except(cur, ri)
        new_max = max(other_max, new_row[-1])
        delta_e = new_max - cur_val

        accept = False
        if delta_e < 0:
            accept = True
        elif delta_e == 0:
            if rng.random() < 0.4:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    accept = True

        if accept:
            old_diffs = cur_row_diffs[ri]
            cur_used -= old_diffs
            new_diffs = row_diffs_set(new_row)
            cur_used |= new_diffs
            cur_row_diffs[ri] = new_diffs

            cur[ri] = new_row
            cur_val = new_max

            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
                best_used = set(cur_used)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T = max(T_min, T * cool)

        # Restart from best if stuck
        if no_improve > 10000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [row_diffs_set(r) for r in cur]
            T = 10.0
            no_improve = 0

        # Check time
        if time.time() > deadline - 5:
            break

    # Final polish: try to minimize each row
    for _ in range(3):
        if time.time() > deadline - 2:
            break
        for ri in range(n_rows):
            if time.time() > deadline - 2:
                break
            forbidden = best_used - row_diffs_set(best[ri])
            new_row = random_valid_row(forbidden, best_val - 1, rng, max_candidates=20)
            if new_row is not None:
                other_max = max_except(best, ri)
                new_max = max(other_max, new_row[-1])
                if new_max < best_val:
                    best[ri] = new_row
                    best_val = new_max
                    best_used = compute_all_used(best)

    # Validate
    if not validate(best):
        best = greedy_init()

    return best

# EVOLVE-BLOCK-END
