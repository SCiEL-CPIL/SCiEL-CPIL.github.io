import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 135s.
    deadline = start_time + 135.0

    # Helper to calculate differences set for a row
    def row_diffs(row):
        out = set()
        n = len(row)
        for i in range(n):
            a = row[i]
            for j in range(i + 1, n):
                out.add(row[j] - a)
        return out

    # Helper to calculate bitmask for a row's differences
    def row_mask(row, max_diff):
        mask = 0
        n = len(row)
        for i in range(n):
            a = row[i]
            for j in range(i + 1, n):
                d = row[j] - a
                if d <= max_diff:
                    mask |= (1 << d)
        return mask

    # Greedy baseline construction
    def greedy_row(used_set, k, cap=None, rng=None):
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

    # Generate baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            row = [0] * (k+1)
        rows.append(row)
        used |= row_diffs(row)

    baseline_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = baseline_max

    # Strategy 1: Precompute and Swap with improved local search
    def generate_valid_rows(limit, k, time_limit):
        """
        Generates valid rows of length k+1 starting at 0, with max element <= limit.
        Returns list of (max_val, mask, row_tuple) or None if timeout.
        """
        start = time.time()
        timeout = time_limit
        rows_list = []

        bit_of = [1 << d for d in range(limit + 1)]

        def backtrack(current_row, current_mask, last_val):
            if time.time() - start > timeout:
                return False

            if len(current_row) == k + 1:
                max_val = current_row[-1]
                rows_list.append((max_val, current_mask, tuple(current_row)))
                return True

            for x in range(last_val + 1, limit + 1):
                valid = True
                new_mask = 0
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

        if backtrack([0], 0, 0):
            return rows_list
        else:
            return None

    # Try to generate pool
    pool = None
    if time.time() < deadline - 60:
        # Generate rows up to baseline_max
        pool = generate_valid_rows(baseline_max, k, 30.0)

        if pool is not None and len(pool) > 0:
            # Sort pool by max_val, then by mask sparsity (fewer bits = more flexible)
            pool.sort(key=lambda x: (x[0], x[1].bit_count()))

            # Initialize state with baseline
            current_rows = [list(r) for r in best_rows]
            current_masks = [row_mask(r, baseline_max) for r in current_rows]

            # Calculate total mask
            total_mask = 0
            for m in current_masks:
                total_mask |= m

            # Improved Local Search: Multiple strategies
            rng = random.Random(42)
            improved = True
            no_improve_count = 0
            max_iterations = 200

            while improved and no_improve_count < max_iterations and time.time() < deadline - 20:
                improved = False
                no_improve_count += 1

                # Strategy A: Replace the row with the largest max
                max_idx = -1
                max_val = -1
                for i, r in enumerate(current_rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_idx = i

                if max_idx >= 0 and max_val > 0:
                    others_mask = total_mask ^ current_masks[max_idx]

                    # Search for best candidate
                    best_cand = None
                    for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                        if cand_max >= max_val:
                            break
                        if (cand_mask & others_mask) == 0:
                            best_cand = (cand_max, cand_mask, cand_row)
                            break

                    if best_cand is not None:
                        cand_max, cand_mask, cand_row = best_cand
                        total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                        current_rows[max_idx] = list(cand_row)
                        current_masks[max_idx] = cand_mask
                        improved = True
                        no_improve_count = 0

                        current_global_max = max(r[-1] for r in current_rows)
                        if current_global_max < best_max:
                            best_max = current_global_max
                            best_rows = [list(r) for r in current_rows]

                # Strategy B: Try swapping any row that can be improved
                if not improved:
                    for i in range(n_rows):
                        if current_rows[i][-1] <= best_max:
                            continue

                        others_mask = total_mask ^ current_masks[i]
                        current_max_i = current_rows[i][-1]

                        best_cand = None
                        for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                            if cand_max >= current_max_i:
                                break
                            if (cand_mask & others_mask) == 0:
                                best_cand = (cand_max, cand_mask, cand_row)
                                break

                        if best_cand is not None:
                            cand_max, cand_mask, cand_row = best_cand
                            total_mask = (total_mask ^ current_masks[i]) | cand_mask
                            current_rows[i] = list(cand_row)
                            current_masks[i] = cand_mask
                            improved = True
                            no_improve_count = 0

                            current_global_max = max(r[-1] for r in current_rows)
                            if current_global_max < best_max:
                                best_max = current_global_max
                                best_rows = [list(r) for r in current_rows]
                            break

                # Strategy C: Random restart of a random row
                if not improved and no_improve_count > 50:
                    j = rng.randint(0, n_rows - 1)
                    others_mask = total_mask ^ current_masks[j]
                    current_max_j = current_rows[j][-1]

                    # Try to find a different valid row for this position
                    candidates = []
                    for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                        if (cand_mask & others_mask) == 0:
                            candidates.append((cand_max, cand_mask, cand_row))

                    if candidates:
                        # Pick a random candidate different from current
                        current_row_tuple = tuple(current_rows[j])
                        different = [c for c in candidates if c[2] != current_row_tuple]
                        if different:
                            cand_max, cand_mask, cand_row = rng.choice(different)
                            total_mask = (total_mask ^ current_masks[j]) | cand_mask
                            current_rows[j] = list(cand_row)
                            current_masks[j] = cand_mask
                            improved = True
                            no_improve_count = 0

                            current_global_max = max(r[-1] for r in current_rows)
                            if current_global_max < best_max:
                                best_max = current_global_max
                                best_rows = [list(r) for r in current_rows]

    # Strategy 2: Targeted DFS polish - try to reduce max by 1, 2, 3...
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

    # Polish phase: try to reduce the global max
    polish_rows = [list(r) for r in best_rows]
    polish_max = best_max
    rng = random.Random(999)

    # Try reducing max by 1, 2, 3
    for reduction in range(1, 4):
        target_max = polish_max - reduction
        if target_max < 0:
            break
        if time.time() > deadline - 10:
            break

        # Try each row in random order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)

        for i in row_order:
            if time.time() > deadline - 10:
                break

            # Skip if this row is already <= target_max
            if polish_rows[i][-1] <= target_max:
                continue

            # Calculate used differences without this row
            used_without_i = set()
            for j in range(n_rows):
                if j != i:
                    used_without_i |= row_diffs(polish_rows[j])

            # Try to find a row that fits within target_max
            row_deadline = min(deadline - 10, time.time() + 5.0)
            cand = dfs_row([0], used_without_i, target_max, row_deadline, randomize=True, rng=rng)

            if cand is not None:
                # Accept if it reduces the global max
                max_others = 0
                for j in range(n_rows):
                    if j != i:
                        if polish_rows[j][-1] > max_others:
                            max_others = polish_rows[j][-1]

                new_global_max = max(max_others, cand[-1])
                if new_global_max < polish_max:
                    polish_rows[i] = cand
                    polish_max = new_global_max
                    best_max = polish_max
                    best_rows = [list(r) for r in polish_rows]
                    break  # Restart with new max

    # Final SA refinement if we have time
    if time.time() < deadline - 15:
        rows_sa = [list(r) for r in best_rows]
        used_sa = set()
        for r in rows_sa:
            used_sa |= row_diffs(r)

        current_max = max(r[-1] for r in rows_sa)
        T = 5.0
        cooling = 0.999
        rng_sa = random.Random(777)
        no_improve = 0

        while time.time() < deadline - 5 and T > 0.1:
            i = rng_sa.randint(0, n_rows - 1)

            used_without_i = set(used_sa)
            used_without_i.difference_update(row_diffs(rows_sa[i]))

            row_deadline = min(deadline - 5, time.time() + 0.3)
            cap = current_max

            cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=True, rng=rng_sa)

            if cand is not None:
                max_others = 0
                for j in range(n_rows):
                    if j != i:
                        if rows_sa[j][-1] > max_others:
                            max_others = rows_sa[j][-1]

                new_global_max = max(max_others, cand[-1])
                delta = new_global_max - current_max

                if delta < 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(-delta / T)
                    except:
                        prob = 0.0
                    accept = rng_sa.random() < prob

                if accept:
                    used_sa.difference_update(row_diffs(rows_sa[i]))
                    rows_sa[i] = cand
                    used_sa |= row_diffs(cand)
                    current_max = new_global_max

                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows_sa]
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve > 50:
                        # Random restart
                        for _ in range(rng_sa.randint(1, 2)):
                            j = rng_sa.randint(0, n_rows - 1)
                            used_sa.difference_update(row_diffs(rows_sa[j]))
                            new_row = dfs_row([0], used_sa, current_max, min(deadline-5, time.time()+0.3), randomize=True, rng=rng_sa)
                            if new_row is not None:
                                rows_sa[j] = new_row
                                used_sa |= row_diffs(new_row)
                                current_max = max(r[-1] for r in rows_sa)
                        no_improve = 0

            T *= cooling

    # Final Validation
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

    if not validate_solution(best_rows):
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            if row is None:
                row = [0] * (k+1)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows
