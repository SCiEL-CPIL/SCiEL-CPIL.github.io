"""
Combined approach: bitmask-accelerated construction with Pareto-front evolutionary
local search for Difference Triangle Set (4,7).

Structural idea from inspiration: use bitmask representation for O(1) difference
conflict checks, and enforce internal distinctness via bit-count validation.
Refinement machinery from current: multi-phase search (greedy init -> backtracking
-> local search with targeted compression and row replacement).
"""
import time
import random
import math
import sys

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin against 150s kill

    def remaining():
        return DEADLINE - (time.time() - t0)

    k_plus_1 = k + 1  # Number of elements per row
    needed = k * (k + 1) // 2  # pairwise differences per row
    total_diffs_needed = n_rows * needed

    def get_diff_mask(row):
        """Calculate the bitmask of all pairwise differences for a row."""
        mask = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Validate rows: strictly increasing, start at 0, all diffs distinct."""
        used_mask = 0
        for r in rows:
            if r[0] != 0:
                return False
            if any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            m = get_diff_mask(r)
            # Check internal distinctness: bit count must equal needed
            if bin(m).count("1") != needed:
                return False
            if m & used_mask:
                return False
            used_mask |= m
        return True

    def sum_of(rows):
        return sum(sum(r) for r in rows)

    # ---- Phase 1: Deterministic greedy baseline ----
    def greedy():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new_diffs_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_diffs_mask |= bit
                if not conflict:
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    # ---- Phase 2: Randomized backtracking constructor ----
    def backtrack_construct(target_max, rng, n_rows, k):
        """Attempt to build n_rows with max entry <= target_max using backtracking."""
        used_mask = 0
        rows = []

        def build_row(used_mask, target_max):
            """Build a single row with backtracking."""
            row = [0]

            def _build(idx, current_row, current_row_mask):
                if idx == k:
                    return current_row[:]

                last = current_row[-1]
                candidates = []
                for x in range(last + 1, target_max + 1):
                    new_diffs_mask = 0
                    conflict = False
                    for a in current_row:
                        d = x - a
                        bit = 1 << d
                        if (used_mask & bit) or (current_row_mask & bit):
                            conflict = True
                            break
                        new_diffs_mask |= bit
                    if not conflict:
                        candidates.append((x, new_diffs_mask))

                if not candidates:
                    return None

                # Randomize for diversity
                rng.shuffle(candidates)

                for x, new_diffs_mask in candidates:
                    res = _build(idx + 1, current_row + [x], current_row_mask | new_diffs_mask)
                    if res is not None:
                        return res
                return None

            return _build(0, row, 0)

        for _ in range(n_rows):
            row = build_row(used_mask, target_max)
            if row is None:
                return None
            rows.append(row)
            used_mask |= get_diff_mask(row)
        return rows

    # ---- Phase 3: Local search with Pareto front ----
    def local_search(start, rng, steps, T0=5.0):
        """Simulated annealing local search to minimize scope."""
        cur = [list(r) for r in start]
        row_masks = [get_diff_mask(r) for r in cur]
        cur_used_mask = 0
        for m in row_masks:
            cur_used_mask |= m

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        cur_sum = sum_of(cur)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj

        # Pareto front: list of (scope, sum, rows_copy)
        pareto = [(cur_scope, cur_sum, [list(r) for r in cur])]

        T = T0
        cooling = 0.999

        for it in range(steps):
            if it % 50 == 0 and remaining() < 8:
                break

            # Select row: bias towards rows with large max
            if rng.random() < 0.5:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)

            row = cur[r]
            other_used_mask = cur_used_mask & ~(row_masks[r])

            move_type = rng.random()
            new_row = None
            new_mask = None

            if move_type < 0.45:
                # MOVE 1: Targeted Compression - try to reduce the max element
                p = len(row) - 1
                current_max = row[p]
                prev_val = row[p - 1]

                for v in range(current_max - 1, prev_val, -1):
                    test_row = row[:]
                    test_row[p] = v
                    if test_row[p - 1] >= v:
                        continue
                    nd_test = get_diff_mask(test_row)
                    if nd_test & other_used_mask:
                        continue
                    if bin(nd_test).count("1") != needed:
                        continue
                    new_row = test_row
                    new_mask = nd_test
                    break

            elif move_type < 0.75:
                # MOVE 2: Swap two elements (not adjacent, to maintain order)
                if len(row) >= 3:
                    i, j = rng.sample(range(1, len(row)), 2)
                    if i > j:
                        i, j = j, i
                    # Swap must maintain increasing order
                    if (i == 1 or row[i - 1] < row[j]) and \
                       (j == len(row) - 1 or row[j + 1] > row[i]) and \
                       (i + 1 == j or row[i + 1] < row[j]) and \
                       (j - 1 == i or row[j - 1] < row[i]):
                        test_row = row[:]
                        test_row[i], test_row[j] = test_row[j], test_row[i]
                        # Re-check strictly increasing
                        valid = all(test_row[m] < test_row[m + 1] for m in range(len(test_row) - 1))
                        if valid:
                            nd_test = get_diff_mask(test_row)
                            if not (nd_test & other_used_mask) and bin(nd_test).count("1") == needed:
                                new_row = test_row
                                new_mask = nd_test

            else:
                # MOVE 3: Random small perturbation - shift one element by small amount
                p = rng.randrange(1, len(row))
                delta = rng.choice([-2, -1, 1, 2])
                new_val = row[p] + delta
                if new_val > row[p - 1] and (p == len(row) - 1 or new_val < row[p + 1]):
                    test_row = row[:]
                    test_row[p] = new_val
                    nd_test = get_diff_mask(test_row)
                    if not (nd_test & other_used_mask) and bin(nd_test).count("1") == needed:
                        new_row = test_row
                        new_mask = nd_test

            if new_row is None:
                continue

            # Evaluate the move
            new_scope = max(max(new_row), max(max(cur[i]) for i in range(n_rows) if i != r))
            new_sum = sum_of([new_row if i == r else cur[i] for i in range(n_rows)])
            new_obj = new_scope * 1000000 + new_sum

            delta_obj = new_obj - cur_obj
            if delta_obj < 0 or rng.random() < math.exp(-delta_obj / max(T, 0.001)):
                # Accept
                cur[r] = new_row
                row_masks[r] = new_mask
                cur_used_mask = cur_used_mask & ~(row_masks[r]) | new_mask
                row_maxes[r] = max(new_row)
                cur_scope = max(row_maxes)
                cur_sum = new_sum
                cur_obj = new_obj

                if cur_obj < best_obj:
                    best = [list(r) for r in cur]
                    best_obj = cur_obj

                # Update Pareto front
                dominated = False
                for ps, psum, _ in pareto:
                    if ps <= cur_scope and psum <= cur_sum and (ps < cur_scope or psum < cur_sum):
                        dominated = True
                        break
                if not dominated:
                    # Remove dominated solutions from front
                    pareto = [(ps, psum, pr) for ps, psum, pr in pareto
                              if not (cur_scope <= ps and cur_sum <= psum and
                                      (cur_scope < ps or cur_sum < psum))]
                    pareto.append((cur_scope, cur_sum, [list(r) for r in cur]))
                    # Cap Pareto front size
                    if len(pareto) > 20:
                        pareto.sort(key=lambda x: x[0] * 1000 + x[1])
                        pareto = pareto[:15]

            T *= cooling

        return best

    # ---- Main search logic ----
    rng = random.Random(12345)

    # Phase 1: Get baseline
    best_rows = greedy()
    best_scope = scope_of(best_rows)

    # Phase 2: Try backtracking with tighter scopes
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))

    for scope_try in range(best_scope - 1, max(lower_bound, 80), -1):
        if remaining() < 15:
            break
        found = False
        for attempt in range(20):
            if remaining() < 15:
                break
            result = backtrack_construct(scope_try, rng, n_rows, k)
            if result is not None and is_valid(result):
                best_rows = result
                best_scope = scope_try
                found = True
                break
        if not found and best_scope - scope_try > 3:
            # If we can't fit in this scope, likely can't in smaller ones
            break

    # Phase 3: Local search from best solution
    if remaining() > 20:
        # Multiple restarts with different seeds
        for restart in range(5):
            if remaining() < 25:
                break
            seed = rng.randint(0, 100000)
            rng2 = random.Random(seed)
            steps = min(5000, int(remaining() * 50))
            result = local_search(best_rows, rng2, steps)
            if is_valid(result):
                s = scope_of(result)
                if s < best_scope:
                    best_rows = result
                    best_scope = s

    # Phase 4: Final targeted compression pass
    if remaining() > 10:
        rows_local = [list(r) for r in best_rows]
        row_masks = [get_diff_mask(r) for r in rows_local]
        improved = True
        while improved and remaining() > 10:
            improved = False
            for r_idx in range(n_rows):
                if remaining() < 10:
                    break
                row = rows_local[r_idx]
                other_mask = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_mask |= row_masks[i]

                # Try to compress each position from the end
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    for new_val in range(row[pos - 1] + 1, old_val):
                        test_row = list(row)
                        test_row[pos] = new_val
                        if not all(test_row[m] < test_row[m + 1] for m in range(len(test_row) - 1)):
                            continue
                        nd = get_diff_mask(test_row)
                        if nd & other_mask:
                            continue
                        if bin(nd).count("1") != needed:
                            continue
                        rows_local[r_idx] = test_row
                        row_masks[r_idx] = nd
                        new_scope = scope_of(rows_local)
                        if new_scope < best_scope:
                            best_rows = [list(r) for r in rows_local]
                            best_scope = new_scope
                        improved = True
                        break

    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy()

    return best_rows
