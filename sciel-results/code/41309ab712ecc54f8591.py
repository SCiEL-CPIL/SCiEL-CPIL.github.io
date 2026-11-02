"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 130
    n_marks = k + 1

    def get_score(rows):
        if rows is None:
            return float('inf')
        return max(max(r) for r in rows)

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in row_diffs or d in all_diffs:
                        return False
                    row_diffs.add(d)
            all_diffs |= row_diffs
        return True

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Phase 1: Best of several greedy constructions
    best_rows = None
    best_score = float('inf')
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        rows = greedy_construct(seed)
        s = get_score(rows)
        if s < best_score:
            best_score = s
            best_rows = [r[:] for r in rows]

    if best_rows is None:
        best_rows = greedy_construct(0)
        best_score = get_score(best_rows)

    # Phase 2: Backtracking to improve individual rows
    def get_used_except_row(rows, skip_idx):
        used = set()
        for i, row in enumerate(rows):
            if i == skip_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    used.add(row[b] - row[a])
        return used

    def try_improve_row(rows, row_idx, time_limit):
        current_row = rows[row_idx]
        current_max = max(current_row)
        other_used = get_used_except_row(rows, row_idx)
        best_row = current_row[:]
        best_max = current_max
        deadline_local = time.time() + time_limit

        def backtrack(pos, current, used_in_row, max_allowed):
            nonlocal best_row, best_max
            if time.time() > deadline_local:
                return
            if pos == n_marks:
                m = max(current)
                if m < best_max:
                    best_max = m
                    best_row = current[:]
                return
            start = current[-1] + 1 if current else 0
            for x in range(start, max_allowed + 1):
                if x >= best_max:
                    break
                new_diffs = {x - a for a in current}
                if new_diffs & used_in_row or new_diffs & other_used:
                    continue
                current.append(x)
                used_in_row |= new_diffs
                backtrack(pos + 1, current, used_in_row, max_allowed)
                current.pop()
                used_in_row -= new_diffs

        # Try to find a row with smaller max
        for target in range(current_max - 1, k, -1):
            if time.time() > deadline_local:
                break
            current = [0]
            used_in_row = set()
            backtrack(1, current, used_in_row, target)
            if best_max < current_max:
                return best_row[:]
        return None

    # Iteratively try to improve rows with largest max
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
        for idx in order:
            if time.time() > deadline - 15:
                break
            row_limit = min(3, deadline - 15 - time.time())
            if row_limit <= 0:
                break
            new_row = try_improve_row(best_rows, idx, row_limit)
            if new_row is not None and max(new_row) < max(best_rows[idx]):
                best_rows[idx] = new_row
                best_score = get_score(best_rows)
                improved = True

    # Phase 3: Random restarts with local search
    if time.time() < deadline - 10:
        current = [r[:] for r in best_rows]
        cur_score = get_score(current)
        best_found = [r[:] for r in current]
        best_found_score = cur_score

        for _ in range(200):
            if time.time() > deadline - 10:
                break
            # Random perturbation: pick a row, rebuild it greedily with random offset
            idx = random.randint(0, n_rows - 1)
            other_used = get_used_except_row(current, idx)
            # Try building row with a random starting bias
            row = [0]
            x = 0
            success = True
            while len(row) < n_marks:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & other_used:
                    row.append(x)
                else:
                    # Try a few more values
                    found = False
                    for x2 in range(x + 1, x + 10):
                        new_diffs2 = {x2 - a for a in row}
                        if not new_diffs2 & other_used:
                            row.append(x2)
                            x = x2
                            found = True
                            break
                    if not found:
                        success = False
                        break
            if success and is_valid(current[:idx] + [row] + current[idx+1:]):
                candidate = current[:idx] + [row] + current[idx+1:]
                s = get_score(candidate)
                if s < best_found_score:
                    best_found_score = s
                    best_found = [r[:] for r in candidate]
                    current = [r[:] for r in candidate]
                    cur_score = s

        if best_found_score < best_score:
            best_rows = [r[:] for r in best_found]
            best_score = best_found_score

    return best_rows
# EVOLVE-BLOCK-END
