"""Difference triangle set solver using Iterated Local Search with multi-start and kick perturbations."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline with randomization ----------
    def greedy(jitter=0.0):
        """Greedy construction with optional random jitter for multi-start."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    if jitter > 0 and random.random() < jitter and x < 200:
                        continue
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    def greedy_with_ordering(order):
        """Greedy construction where rows are built in a specific order."""
        used = set()
        rows = [None] * n_rows
        for idx in order:
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows[idx] = row
        return rows

    # ---------- Local search: swap-based improvement ----------
    def local_search(rows, max_iters=500):
        """Local search by trying to replace individual entries in rows."""
        rows = [list(r) for r in rows]
        improved = True
        iters = 0
        while improved and time_left() > 10 and iters < max_iters:
            improved = False
            iters += 1
            # Try each row in random order
            for row_idx in random.sample(range(n_rows), n_rows):
                if time_left() < 10:
                    break
                row = rows[row_idx]
                # Try replacing each non-zero entry with a smaller valid value
                for pos in range(1, len(row)):
                    if time_left() < 10:
                        break
                    old_val = row[pos]
                    # Try smaller values
                    for new_val in range(1, old_val):
                        candidate = list(row)
                        candidate[pos] = new_val
                        # Check if still increasing
                        if pos > 0 and new_val <= candidate[pos-1]:
                            continue
                        if pos < len(candidate)-1 and new_val >= candidate[pos+1]:
                            continue
                        # Check all differences are valid
                        if _row_is_valid_in_context(candidate, rows, row_idx):
                            rows[row_idx] = candidate
                            improved = True
                            break
        return rows

    def _row_is_valid_in_context(candidate_row, all_rows, row_idx):
        """Check if candidate_row has no internal duplicates and no conflicts with other rows."""
        # Internal duplicates
        diffs = set()
        for i in range(len(candidate_row)):
            for j in range(i+1, len(candidate_row)):
                d = candidate_row[j] - candidate_row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        # Cross-row conflicts
        for other_idx in range(n_rows):
            if other_idx == row_idx:
                continue
            other = all_rows[other_idx]
            for i in range(len(other)):
                for j in range(i+1, len(other)):
                    d = other[j] - other[i]
                    if d in diffs:
                        return False
        return True

    # ---------- Kick perturbation for ILS ----------
    def kick(rows, strength=1):
        """Perturb the solution by randomly modifying a few entries."""
        rows = [list(r) for r in rows]
        n_kicks = strength
        for _ in range(n_kicks):
            if time_left() < 5:
                break
            row_idx = random.randrange(n_rows)
            pos = random.randrange(1, len(rows[row_idx]))
            old_val = rows[row_idx][pos]
            # Try to move this entry to a random nearby position
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = old_val + delta
            if new_val < 1:
                continue
            candidate = list(rows[row_idx])
            candidate[pos] = new_val
            # Check increasing constraint
            if pos > 0 and new_val <= candidate[pos-1]:
                continue
            if pos < len(candidate)-1 and new_val >= candidate[pos+1]:
                continue
            if _row_is_valid_in_context(candidate, rows, row_idx):
                rows[row_idx] = candidate
        return rows

    # ---------- Multi-start ILS ----------
    best_rows = greedy()
    best_scope = scope(best_rows)

    # Try several different greedy orderings
    for trial in range(5):
        if time_left() < 20:
            break
        order = list(range(n_rows))
        random.shuffle(order)
        candidate_rows = greedy_with_ordering(order)
        candidate_scope = scope(candidate_rows)
        if candidate_scope < best_scope:
            best_rows = candidate_rows
            best_scope = candidate_scope

    # ILS loop
    current_rows = list(best_rows)
    current_scope = best_scope
    stagnation = 0
    max_stagnation = 50

    while time_left() > 15:
        # Local search
        ls_rows = local_search(current_rows, max_iters=200)
        ls_scope = scope(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
            current_rows = ls_rows
            current_scope = ls_scope
            stagnation = 0
        else:
            stagnation += 1

        if stagnation > max_stagnation:
            # Restart with a new random greedy
            if time_left() < 20:
                break
            new_rows = greedy(jitter=random.uniform(0.05, 0.15))
            new_scope = scope(new_rows)
            if new_scope < current_scope:
                current_rows = new_rows
                current_scope = new_scope
            stagnation = 0
            continue

        # Kick perturbation
        if time_left() < 10:
            break
        strength = min(3, 1 + stagnation // 10)
        kicked_rows = kick(current_rows, strength=strength)
        kicked_scope = scope(kicked_rows)
        # Accept if better or with some probability (to escape local minima)
        if kicked_scope <= current_scope or (random.random() < 0.3 and kicked_scope < current_scope + 5):
            current_rows = kicked_rows
            current_scope = kicked_scope

    # Final polish
    if time_left() > 10:
        polished = local_search(best_rows, max_iters=300)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(best_rows)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
