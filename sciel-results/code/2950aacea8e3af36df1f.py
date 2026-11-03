"""Difference triangle set (4,7): construct rows with distinct global differences.

Combined approach:
1. Fast greedy baseline.
2. Aggressive "pull-to-smallest" local search (structural from inspiration).
3. Constrained backtracking to hit a target scope (structural from inspiration).
4. Randomized restarts to escape local minima.
"""
import time
import random
import sys


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    k_plus_1 = k + 1
    diffs_per_row = k_plus_1 * k // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope
    lb = total_diffs - 1

    # 1. Fast greedy baseline
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)

    if best_scope <= lb:
        return best_rows

    # 2. Aggressive local search: pull-to-smallest
    def _improve_scope(rows, max_iters=500000):
        if not rows:
            return rows

        best = [list(row) for row in rows]
        best_scope_ls = max(max(row) for row in best)

        used_diffs = set()
        for row in best:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used_diffs.add(row[j] - row[i])

        def _pull_to_smallest(row_idx):
            row = best[row_idx]
            any_changed = False
            for pos in range(1, k_plus_1):
                # Find smallest feasible value > row[pos-1]
                found = False
                for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                    new_diffs = set()
                    ok = True
                    # Check diffs with previous elements
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    # Check diffs with next elements
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    # Check no duplicate within the new set of diffs for this row
                    # Actually, since we are changing one element, we need to ensure
                    # the new diffs don't conflict with each other?
                    # The diffs are {new_val - row[i] for i < pos} U {row[j] - new_val for j > pos}
                    # These are all distinct if the row is strictly increasing?
                    # Yes, because row[i] < new_val < row[j] implies new_val - row[i] > 0 and row[j] - new_val > 0.
                    # Could new_val - row[i] == row[j] - new_val?
                    # 2*new_val == row[i] + row[j]. Possible.
                    # So we must check uniqueness within new_diffs.
                    if len(new_diffs) != len(row) - 1:
                        continue

                    # Remove old diffs involving pos
                    old_diffs = set()
                    for j in range(k_plus_1):
                        if j != pos:
                            d = abs(row[pos] - row[j])
                            old_diffs.add(d)
                    for d in old_diffs:
                        used_diffs.discard(d)

                    # Add new diffs
                    used_diffs |= new_diffs
                    row[pos] = new_val
                    any_changed = True
                    found = True
                    break
                if found:
                    continue
                # If we didn't find a smaller value, keep original
            return any_changed

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            changed = False
            # Try pulling each row to smallest
            for ridx in range(n_rows):
                if _pull_to_smallest(ridx):
                    changed = True

            if not changed:
                break

            current_scope = max(max(r) for r in best)
            if current_scope < best_scope_ls:
                best_scope_ls = current_scope

        return best, best_scope_ls

    improved_rows, improved_scope = _improve_scope(best_rows)
    if improved_scope < best_scope:
        best_scope = improved_scope
        best_rows = improved_rows

    if best_scope <= lb:
        return best_rows

    # 3. Backtracking search to hit a target scope
    def _backtracking_search(target_scope, seed_offset=0):
        rng = random.Random(seed_offset * 12345 + 678)
        rows = []
        used_diffs = [False] * (target_scope + 1)

        def build_row(pos, current_mark, row_so_far):
            if time.time() > deadline:
                return None

            if pos == k_plus_1:
                return list(row_so_far)

            max_x = target_scope - (k_plus_1 - pos - 1)
            if max_x <= current_mark:
                return None

            valid_candidates = []
            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d > target_scope or used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                if len(valid_candidates) >= 50:
                    break

            if not valid_candidates:
                return None

            n_smallest = min(15, len(valid_candidates))
            candidates_to_try = valid_candidates[:n_smallest]

            if len(valid_candidates) > n_smallest and rng.random() < 0.25:
                extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
                candidates_to_try = candidates_to_try + extra

            rng.shuffle(candidates_to_try)

            for x in candidates_to_try:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True

                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res

                for d in new_diffs:
                    used_diffs[d] = False

            return None

        for i in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for j in range(k_plus_1):
                for i2 in range(j + 1, k_plus_1):
                    d = row[i2] - row[j]
                    if not used_diffs[d]:
                        used_diffs[d] = True

        return rows

    # 4. Iterative deepening backtracking + local search
    target = best_scope - 1
    attempts = 0
    while time.time() < deadline - 10.0 and target > lb:
        found = False
        for _ in range(20):  # Limit restarts per target
            if time.time() > deadline - 10.0:
                break
            sol = _backtracking_search(target, attempts)
            if sol:
                new_scope = max(max(r) for r in sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = sol
                    found = True
                    # Improve further
                    improved_rows, improved_scope = _improve_scope(best_rows, max_iters=100000)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_rows = improved_rows
                    if best_scope <= lb:
                        return best_rows
                    break
            attempts += 1

        if not found:
            target -= 1
        else:
            target = best_scope - 1

    return best_rows


if __name__ == "__main__":
    rows = construct(4, 7)
    for r in rows:
        print(r)
    print("Scope:", max(max(r) for r in rows))
