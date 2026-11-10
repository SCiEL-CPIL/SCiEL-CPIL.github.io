import time
import random
import math
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 145  # Safety margin before 150s limit

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None, randomize=False, rng=None):
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline, randomize=False, rng=None):
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None

        candidates = list(range(last + 1, upper + 1))
        if randomize and rng is not None:
            rng.shuffle(candidates)

        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize, rng)
            if res is not None:
                return res
        return None

    def validate_solution(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    # Phase 1: Greedy Baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            # Fallback: try with randomization
            row = greedy_row(set(used), k, randomize=True)
        if row is None:
            # Should not happen for reasonable inputs, but be safe
            return [[0] * (k+1) for _ in range(n_rows)]
        rows.append(row)
        used |= row_diffs(row)

    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max
    row_diff_sets = [row_diffs(r) for r in rows]

    # Phase 2: Simulated Annealing with improved candidate generation
    T = 20.0
    cooling = 0.995
    rng = random.Random(42)

    def generate_candidate(i, cap=None, randomize=True):
        if cap is None:
            cap = current_max
        # Calculate used differences excluding row i
        used_without_i = set(used)
        used_without_i.difference_update(row_diff_sets[i])

        row_deadline = min(deadline - 5, time.time() + 0.5)
        cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=randomize, rng=rng)
        return cand

    # Main SA Loop
    no_improve_count = 0
    sa_start = time.time()
    while time.time() < deadline - 20:
        if T < 0.05:
            break

        # Pick a random row to perturb
        i = rng.randint(0, n_rows - 1)

        # Try to generate a candidate
        # Sometimes allow slightly larger cap to explore
        cap = current_max
        if rng.random() < 0.15:
            cap = current_max + rng.randint(1, 8)

        cand = generate_candidate(i, cap=cap, randomize=True)

        if cand is not None:
            new_last = cand[-1]
            max_others = 0
            for j in range(n_rows):
                if j != i:
                    if rows[j][-1] > max_others:
                        max_others = rows[j][-1]

            new_global_max = max(max_others, new_last)
            delta = new_global_max - current_max

            if delta < 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                accept = rng.random() < prob

            if accept:
                # Update state
                used.difference_update(row_diff_sets[i])
                rows[i] = cand
                row_diff_sets[i] = row_diffs(cand)
                used |= row_diff_sets[i]

                current_max = new_global_max

                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # If stuck for too long, restart with a different seed or perturb more
                if no_improve_count > 300:
                    no_improve_count = 0
                    # Randomly reset a few rows to break symmetry
                    for _ in range(rng.randint(1, 3)):
                        j = rng.randint(0, n_rows - 1)
                        used.difference_update(row_diff_sets[j])
                        new_row = greedy_row(set(used), k, randomize=True, rng=rng)
                        if new_row is not None:
                            rows[j] = new_row
                            row_diff_sets[j] = row_diffs(new_row)
                            used |= row_diff_sets[j]
                            current_max = max(r[-1] for r in rows)

        T *= cooling

    # Phase 3: Local Search - try to minimize each row's last element
    final_rows = best_rows[:]
    final_used = set()
    final_row_diffs = []
    for r in final_rows:
        d = row_diffs(r)
        final_used |= d
        final_row_diffs.append(d)

    # Try to shrink each row further using iterative deepening
    for i in range(n_rows):
        if time.time() > deadline - 10:
            break
        cur_last = final_rows[i][-1]
        if cur_last == 0:
            continue

        used_without = set(final_used)
        used_without.difference_update(final_row_diffs[i])

        # Try to find a smaller row using binary search on cap
        found = None
        low_cap = 0
        high_cap = cur_last - 1

        # First check if we can even fit in high_cap
        row_deadline = min(deadline - 10, time.time() + 1.5)
        test_cand = dfs_row([0], used_without, high_cap, row_deadline, randomize=False, rng=rng)
        if test_cand is None:
            continue

        # Binary search for the minimum cap
        while low_cap <= high_cap and time.time() < deadline - 10:
            mid_cap = (low_cap + high_cap) // 2
            row_deadline = min(deadline - 10, time.time() + 1.0)
            cand = dfs_row([0], used_without, mid_cap, row_deadline, randomize=False, rng=rng)
            if cand is not None:
                found = cand
                high_cap = mid_cap - 1
            else:
                low_cap = mid_cap + 1

        if found is not None:
            final_used.difference_update(final_row_diffs[i])
            final_rows[i] = found
            final_row_diffs[i] = row_diffs(found)
            final_used |= final_row_diffs[i]

    # Phase 4: Final polish - try pairwise swaps to reduce max
    if time.time() < deadline - 5:
        for i in range(n_rows):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n_rows):
                if time.time() > deadline - 5:
                    break
                # Try swapping rows i and j
                old_i = final_rows[i]
                old_j = final_rows[j]
                
                # Check if swapping helps
                max_without_i = max(r[-1] for r in final_rows if r is not old_i)
                max_without_j = max(r[-1] for r in final_rows if r is not old_j)
                
                # Try to regenerate row i with the space freed by row j
                used_without_i = set(final_used)
                used_without_i.difference_update(final_row_diffs[i])
                
                cap_i = min(old_i[-1], max_without_j)
                row_deadline = min(deadline - 5, time.time() + 0.3)
                new_i = dfs_row([0], used_without_i, cap_i, row_deadline, randomize=False, rng=rng)
                
                if new_i is not None and new_i[-1] < old_i[-1]:
                    final_used.difference_update(final_row_diffs[i])
                    final_rows[i] = new_i
                    final_row_diffs[i] = row_diffs(new_i)
                    final_used |= final_row_diffs[i]

    # Final validation
    if not validate_solution(final_rows):
        # Fallback to best_rows if final polish broke something (shouldn't happen)
        if validate_solution(best_rows):
            return best_rows
        # Last resort: return greedy baseline
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return final_rows
