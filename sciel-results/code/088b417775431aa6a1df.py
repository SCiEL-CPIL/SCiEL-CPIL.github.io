import time
import random
import math
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135  # Safety margin before 150s limit

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
                if row[i] >= row[i + 1]:
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
            row = greedy_row(set(used), k, randomize=True)
        if row is None:
            return [[0] * (k + 1) for _ in range(n_rows)]
        rows.append(row)
        used |= row_diffs(row)

    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max

    # Phase 2: Precompute pool of all valid rows up to a cap
    # This enables fast O(1) bitmask validity checks during local search
    pool_cap = current_max + 5  # Slightly above baseline to allow exploration
    pool_deadline = min(deadline - 60, start + 25)  # Max 25s for pool generation

    def generate_pool(limit, time_limit):
        """Generate all valid rows of length k+1 starting at 0 with max <= limit."""
        start_t = time.time()
        timeout = time_limit
        pool = []  # list of (max_val, frozenset_of_diffs, row_tuple)

        bit_of = [1 << d for d in range(limit + 1)]

        def backtrack(current_row, current_mask, last_val):
            if time.time() - start_t > timeout:
                return False
            if len(current_row) == k + 1:
                max_val = current_row[-1]
                # Compute diff set
                diffs = frozenset()
                for i in range(len(current_row)):
                    a = current_row[i]
                    for j in range(i + 1, len(current_row)):
                        diffs = diffs | frozenset([current_row[j] - a])
                pool.append((max_val, current_mask, tuple(current_row)))
                return True

            for x in range(last_val + 1, limit + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > limit:
                        valid = False
                        break
                    bit = bit_of[d]
                    if current_mask & bit:
                        valid = False
                        break
                    new_mask |= bit

                if valid:
                    current_row.append(x)
                    if not backtrack(current_row, current_mask | new_mask, x):
                        return False
                    current_row.pop()
            return True

        if backtrack([0], 0, 0):
            return pool
        return None

    pool = None
    if time.time() < pool_deadline:
        pool = generate_pool(pool_cap, pool_deadline - start)

    # Phase 3: Local search using pool (fast bitmask checks)
    if pool is not None and len(pool) > 0:
        # Sort pool by max_val for early termination
        pool.sort(key=lambda x: x[0])

        current_rows = [list(r) for r in best_rows]
        current_masks = []
        for r in current_rows:
            # Compute mask for this row
            mask = 0
            for d in row_diffs(r):
                if d <= pool_cap:
                    mask |= (1 << d)
            current_masks.append(mask)

        total_mask = 0
        for m in current_masks:
            total_mask |= m

        rng = random.Random(42)
        improved = True
        no_improve_count = 0

        while improved and time.time() < deadline - 15:
            improved = False
            no_improve_count += 1
            if no_improve_count > 150:
                break

            # Find row with max value
            max_idx = -1
            max_val = -1
            for i, r in enumerate(current_rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_idx = i

            if max_idx == -1:
                break

            # Mask of all other rows
            others_mask = total_mask ^ current_masks[max_idx]

            # Search pool for a valid candidate with smaller max
            found = False
            for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                if cand_max >= max_val:
                    break
                if (cand_mask & others_mask) == 0:
                    # Valid swap
                    total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                    current_rows[max_idx] = list(cand_row)
                    current_masks[max_idx] = cand_mask
                    improved = True
                    found = True
                    break

            if not found:
                # Try randomized swap: pick a random row and try to replace with any valid smaller row
                i = rng.randint(0, n_rows - 1)
                cur_max_i = current_rows[i][-1]
                others_mask_i = total_mask ^ current_masks[i]

                # Collect all valid candidates for this row
                valid_cands = []
                for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                    if cand_max >= cur_max_i:
                        break
                    if (cand_mask & others_mask_i) == 0:
                        valid_cands.append((cand_max, cand_mask, cand_row))

                if valid_cands:
                    # Pick the best (smallest max) or random
                    if rng.random() < 0.7:
                        choice = valid_cands[0]  # Best
                    else:
                        choice = rng.choice(valid_cands)

                    total_mask = (total_mask ^ current_masks[i]) | choice[1]
                    current_rows[i] = list(choice[2])
                    current_masks[i] = choice[1]
                    improved = True

            # Update best
            current_global_max = max(r[-1] for r in current_rows)
            if current_global_max < best_max:
                best_max = current_global_max
                best_rows = [list(r) for r in current_rows]

    # Phase 4: Simulated Annealing refinement (on best_rows)
    T = 25.0
    cooling = 0.993
    rng = random.Random(123)

    current_rows = [list(r) for r in best_rows]
    current_max = max(r[-1] for r in current_rows)
    row_diff_sets = [row_diffs(r) for r in current_rows]
    used = set()
    for d in row_diff_sets:
        used |= d

    def generate_candidate(i, cap=None, randomize=True):
        if cap is None:
            cap = current_max
        used_without_i = set(used)
        used_without_i.difference_update(row_diff_sets[i])
        row_deadline = min(deadline - 10, time.time() + 0.3)
        cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=randomize, rng=rng)
        return cand

    no_improve_count = 0
    while time.time() < deadline - 20:
        if T < 0.05:
            break

        i = rng.randint(0, n_rows - 1)
        cap = current_max
        if rng.random() < 0.2:
            cap = current_max + rng.randint(1, 10)

        cand = generate_candidate(i, cap=cap, randomize=True)

        if cand is not None:
            new_last = cand[-1]
            max_others = 0
            for j in range(n_rows):
                if j != i and current_rows[j][-1] > max_others:
                    max_others = current_rows[j][-1]

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
                used.difference_update(row_diff_sets[i])
                current_rows[i] = cand
                row_diff_sets[i] = row_diffs(cand)
                used |= row_diff_sets[i]
                current_max = new_global_max

                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > 200:
                    no_improve_count = 0
                    for _ in range(rng.randint(2, 3)):
                        j = rng.randint(0, n_rows - 1)
                        used.difference_update(row_diff_sets[j])
                        new_row = greedy_row(set(used), k, randomize=True, rng=rng)
                        if new_row is not None:
                            current_rows[j] = new_row
                            row_diff_sets[j] = row_diffs(new_row)
                            used |= row_diff_sets[j]
                            current_max = max(r[-1] for r in current_rows)

        T *= cooling

    # Phase 5: Intensive local search on best_rows
    final_rows = [list(r) for r in best_rows]
    final_used = set()
    final_row_diffs = []
    for r in final_rows:
        d = row_diffs(r)
        final_used |= d
        final_row_diffs.append(d)

    # 5a: Try to shrink each row via binary search on cap
    for i in range(n_rows):
        if time.time() > deadline - 10:
            break
        cur_last = final_rows[i][-1]
        if cur_last == 0:
            continue

        used_without = set(final_used)
        used_without.difference_update(final_row_diffs[i])

        found = None
        low_cap = 0
        high_cap = cur_last - 1

        row_deadline = min(deadline - 10, time.time() + 0.8)
        test_cand = dfs_row([0], used_without, high_cap, row_deadline, randomize=False, rng=rng)
        if test_cand is None:
            continue

        while low_cap <= high_cap and time.time() < deadline - 10:
            mid_cap = (low_cap + high_cap) // 2
            row_deadline = min(deadline - 10, time.time() + 0.6)
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

    # 5b: Pairwise row swap attempts
    if time.time() < deadline - 5:
        for i in range(n_rows):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n_rows):
                if time.time() > deadline - 5:
                    break
                old_i = final_rows[i]
                old_j = final_rows[j]

                combined_free = set(final_used)
                combined_free.difference_update(final_row_diffs[i])
                combined_free.difference_update(final_row_diffs[j])

                cap_i = min(old_i[-1], old_j[-1])
                row_deadline = min(deadline - 5, time.time() + 0.2)
                new_i = dfs_row([0], combined_free, cap_i, row_deadline, randomize=False, rng=rng)

                if new_i is not None and new_i[-1] < old_i[-1]:
                    final_used.difference_update(final_row_diffs[i])
                    final_rows[i] = new_i
                    final_row_diffs[i] = row_diffs(new_i)
                    final_used |= final_row_diffs[i]

    # 5c: Final tight polish
    if time.time() < deadline - 2:
        current_global = max(r[-1] for r in final_rows)
        for i in range(n_rows):
            if time.time() > deadline - 2:
                break
            if final_rows[i][-1] <= current_global - 1:
                continue
            used_without = set(final_used)
            used_without.difference_update(final_row_diffs[i])
            row_deadline = min(deadline - 2, time.time() + 0.4)
            cand = dfs_row([0], used_without, current_global - 1, row_deadline, randomize=False, rng=rng)
            if cand is not None:
                final_used.difference_update(final_row_diffs[i])
                final_rows[i] = cand
                final_row_diffs[i] = row_diffs(cand)
                final_used |= final_row_diffs[i]

    # Final validation
    if not validate_solution(final_rows):
        if validate_solution(best_rows):
            return best_rows
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return final_rows
