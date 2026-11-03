import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 118.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs

    # ---- Bitset helpers ----
    def diff_mask(row):
        mask = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---- Fast greedy initial solution ----
    def greedy_initial(rng, cap=500):
        used_mask = 0
        rows = []
        for r in range(n_rows):
            row = [0]
            cur_mask = 0
            x = 1
            while len(row) < m and x < cap:
                new_d_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    new_d_mask |= (1 << d)
                if (used_mask | cur_mask) & new_d_mask:
                    x += 1
                    continue
                row.append(x)
                cur_mask |= new_d_mask
                x += 1
            if len(row) < m:
                return None
            rows.append(row)
            used_mask |= cur_mask
        return rows

    # ---- Backtracking row generation with bitset pruning ----
    def backtrack_row(used_mask, max_val, rng, cap=300):
        row = [0]
        cur_mask = 0

        def dfs(pos, last_val):
            nonlocal row, cur_mask
            if time.time() > deadline - 5:
                return False
            if pos == k:
                return True
            rem = k - pos - 1
            min_next = last_val + 1
            max_next = max_val - rem
            if min_next > max_next:
                return False

            candidates = []
            for x in range(min_next, max_next + 1):
                new_d_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    new_d_mask |= (1 << d)
                if (used_mask | cur_mask) & new_d_mask:
                    continue
                candidates.append((x, new_d_mask))
                if len(candidates) > 40:
                    break

            if not candidates:
                return False

            candidates.sort(key=lambda t: t[0])
            if rng.random() < 0.7:
                trial_list = candidates[:6]
            else:
                trial_list = rng.sample(candidates, min(10, len(candidates)))

            for x, new_mask in trial_list:
                row.append(x)
                old_cur_mask = cur_mask
                cur_mask |= new_mask
                if dfs(pos + 1, x):
                    return True
                else:
                    row.pop()
                    cur_mask = old_cur_mask
            return False

        if dfs(0, 0):
            return row, cur_mask
        return None, 0

    # ---- Simulated annealing with row regeneration ----
    def anneal(rows, rng, max_iter=30000):
        rows = [list(r) for r in rows]
        row_masks = [diff_mask(r) for r in rows]
        used_mask = 0
        for rm in row_masks:
            used_mask |= rm

        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 5.0
        cooling = 0.9995
        min_T = 0.1

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_mask = row_masks[r_idx]
            used_mask &= ~old_mask

            cap = int(best_max * 1.15) + 10
            new_row, new_mask = backtrack_row(used_mask, cap, rng)

            if new_row is None:
                used_mask |= old_mask
                row_masks[r_idx] = old_mask
                continue

            new_global = max(max(rows[i][-1] for i in range(n_rows) if i != r_idx), new_row[-1])
            delta = new_global - current_max

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    accept = True

            if accept:
                rows[r_idx] = new_row
                row_masks[r_idx] = new_mask
                used_mask |= new_mask
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                used_mask |= old_mask
                row_masks[r_idx] = old_mask

            T = max(T * cooling, min_T)

        return best_rows, best_max

    # ---- Gap-based local search with cross-row gap exchange ----
    def gaps_from_row(row):
        return [row[i + 1] - row[i] for i in range(len(row) - 1)]

    def row_from_gaps(gaps):
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        return row

    def gap_exchange_search(rows, rng, deadline_local):
        """Multi-start local search optimizing global max via gap-exchange moves between rows."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_scope = scope_of(rows)

        # Precompute diff masks
        row_masks = [diff_mask(r) for r in rows]
        used_mask = 0
        for rm in row_masks:
            used_mask |= rm

        # Local search: try exchanging gaps between two rows
        for iteration in range(5000):
            if time.time() > deadline_local:
                break

            # Pick two random rows
            if n_rows < 2:
                break
            r1, r2 = rng.sample(range(n_rows), 2)

            g1 = gaps_from_row(rows[r1])
            g2 = gaps_from_row(rows[r2])

            # Pick random gap positions in each row
            p1 = rng.randint(0, k - 1)
            p2 = rng.randint(0, k - 1)

            old_g1 = g1[p1]
            old_g2 = g2[p2]

            # Try swapping
            g1[p1] = old_g2
            g2[p2] = old_g1

            new_row1 = row_from_gaps(g1)
            new_row2 = row_from_gaps(g2)

            # Check validity: strictly increasing (gaps > 0) and no diff conflicts
            if any(g <= 0 for g in g1) or any(g <= 0 for g in g2):
                g1[p1] = old_g1
                g2[p2] = old_g2
                continue

            # Check diff conflicts
            new_mask1 = diff_mask(new_row1)
            new_mask2 = diff_mask(new_row2)

            # Remove old masks, add new
            temp_used = (used_mask & ~row_masks[r1] & ~row_masks[r2]) | new_mask1 | new_mask2
            if temp_used == used_mask | new_mask1 | new_mask2:
                # No conflict (new masks don't overlap with other rows' masks)
                pass
            else:
                # Check if new masks conflict with each other or with remaining used
                other_used = used_mask & ~row_masks[r1] & ~row_masks[r2]
                if (new_mask1 & new_mask2) or (new_mask1 & other_used) or (new_mask2 & other_used):
                    g1[p1] = old_g1
                    g2[p2] = old_g2
                    continue

            # Accept if it improves or maintains scope
            new_scope = max(new_row1[-1], new_row2[-1],
                           max(rows[i][-1] for i in range(n_rows) if i not in (r1, r2)))
            current_scope = scope_of(rows)

            if new_scope <= current_scope:
                # Apply
                used_mask = (used_mask & ~row_masks[r1] & ~row_masks[r2]) | new_mask1 | new_mask2
                rows[r1] = new_row1
                rows[r2] = new_row2
                row_masks[r1] = new_mask1
                row_masks[r2] = new_mask2

                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Revert
                g1[p1] = old_g1
                g2[p2] = old_g2

        return best_rows, best_scope

    # ---- Element shift local search (from current program) ----
    def improve_scope(solution, deadline_local):
        if not solution:
            return solution
        best = [list(row) for row in solution]
        k_plus_1 = k + 1

        used_diffs = set()
        for row in best:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used_diffs.add(row[j] - row[i])

        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            for row_idx in range(n_rows):
                row = best[row_idx]
                for pos in range(k_plus_1 - 1, 0, -1):
                    original = row[pos]
                    for new_val in range(row[pos - 1] + 1, original):
                        new_diffs = {new_val - row[i] for i in range(pos)}
                        for j in range(pos + 1, k_plus_1):
                            new_diffs.add(row[j] - new_val)
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(original - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - original)
                        if new_diffs & (used_diffs - old_diffs):
                            continue
                        if any(d <= 0 for d in new_diffs):
                            continue
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        improved = True
                        break

        return best

    # ---- Main pipeline ----
    rng = random.Random(42)

    # 1. Greedy baseline
    best_solution = greedy_initial(rng)
    if not best_solution:
        return []
    best_scope = scope_of(best_solution)

    if best_scope <= lb:
        return best_solution

    # 2. Simulated annealing with row regeneration
    anneal_deadline = deadline - 30
    best_solution, best_scope = anneal(best_solution, rng, max_iter=20000)
    if best_scope <= lb:
        return best_solution

    # 3. Element shift improvement
    shift_deadline = deadline - 15
    best_solution = improve_scope(best_solution, shift_deadline)
    best_scope = scope_of(best_solution)
    if best_scope <= lb:
        return best_solution

    # 4. Gap-exchange local search (structural reconfiguration)
    gap_deadline = deadline - 5
    best_solution, best_scope = gap_exchange_search(best_solution, rng, gap_deadline)
    if best_scope <= lb:
        return best_solution

    # 5. Final polish
    best_solution = improve_scope(best_solution, deadline - 1)
    best_scope = scope_of(best_solution)

    return best_solution
