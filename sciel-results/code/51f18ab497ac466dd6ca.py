import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    # Lower bound: need n_rows * k*(k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_build(rng):
        """Build rows greedily, trying to keep them compact."""
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                # Try to find the smallest valid next element
                # Search in a window, with randomization to escape local traps
                found = False
                # First try in order (compact), then randomized
                base_search = last + 1
                max_search = last + 200
                # Occasionally jump ahead
                if rng.random() < 0.3:
                    start_offset = rng.randint(1, 30)
                else:
                    start_offset = 1

                # Try a batch of candidates
                candidates = []
                for x in range(base_search + start_offset - 1, min(base_search + start_offset - 1 + max_search, lb + 300)):
                    candidates.append(x)
                # Shuffle a portion for diversity
                if rng.random() < 0.5:
                    rng.shuffle(candidates)
                else:
                    # Mostly ordered but with some perturbation
                    candidates = sorted(candidates, key=lambda x: x + rng.random() * 10)

                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d <= 0:
                            valid = False
                            break
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        for d in new_diffs:
                            used.add(d)
                        found = True
                        break

                if not found:
                    # Wider search
                    for x in range(last + 1, lb + 500):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d <= 0:
                                valid = False
                                break
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            row.append(x)
                            for d in new_diffs:
                                used.add(d)
                            found = True
                            break
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_search(rows, rng, time_limit):
        """Improve rows by trying to reduce the maximum entry.
        Strategy: for each row, try to move elements to smaller positions.
        Also try removing and re-adding elements."""
        if rows is None:
            return None, float('inf')

        used = compute_used(rows)
        best_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]

        # Try to improve by reducing elements
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Randomize row order
            row_order = list(range(n_rows))
            rng.shuffle(row_order)

            for row_idx in row_order:
                row = best_rows[row_idx]
                # Try to reduce each element (from last to first to maintain order)
                for pos in range(len(row) - 1, 0, -1):
                    if row[pos] <= row[pos - 1] + 1:
                        continue  # Already minimal
                    # Try to reduce row[pos]
                    current_val = row[pos]
                    # Remove its differences
                    diffs_to_remove = set()
                    for i in range(pos):
                        diffs_to_remove.add(current_val - row[i])
                    for d in diffs_to_remove:
                        used.discard(d)

                    # Try smaller values
                    new_val = None
                    search_start = row[pos - 1] + 1
                    search_end = current_val
                    # Try in random order with bias toward smaller
                    candidates = list(range(search_start, search_end))
                    if candidates:
                        # Bias toward smaller values
                        if rng.random() < 0.7:
                            # Mostly try smaller first
                            candidates.sort(key=lambda x: x + rng.random() * 5)
                        else:
                            rng.shuffle(candidates)

                        for x in candidates:
                            valid = True
                            new_diffs = set()
                            for i in range(pos):
                                d = x - row[i]
                                if d <= 0:
                                    valid = False
                                    break
                                if d in used:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                new_val = x
                                break

                    if new_val is not None and new_val < current_val:
                        # Accept the move
                        row[pos] = new_val
                        for i in range(pos):
                            used.add(new_val - row[i])
                        improved = True
                        best_scope = max(max(r) for r in best_rows)
                    else:
                        # Restore differences
                        for d in diffs_to_remove:
                            used.add(d)

                # Occasionally try a more aggressive move: remove an element and re-add
                if rng.random() < 0.3 and len(row) > 1:
                    pos = rng.randint(1, len(row) - 1)
                    val = row[pos]
                    # Remove
                    diffs_to_remove = set()
                    for i in range(pos):
                        diffs_to_remove.add(val - row[i])
                    for d in diffs_to_remove:
                        used.discard(d)
                    row.pop(pos)

                    # Re-add at a smaller position
                    new_val = None
                    # Find where to insert
                    for insert_pos in range(len(row), 0, -1):
                        # Check if we can insert at insert_pos
                        prev = row[insert_pos - 1] if insert_pos > 0 else 0
                        next_val = row[insert_pos] if insert_pos < len(row) else float('inf')
                        search_start = prev + 1
                        search_end = min(next_val, val)
                        if search_start >= search_end:
                            continue
                        candidates = list(range(search_start, search_end))
                        rng.shuffle(candidates)
                        for x in candidates:
                            valid = True
                            new_diffs = set()
                            for i in range(len(row)):
                                if i < insert_pos:
                                    d = x - row[i]
                                    if d <= 0:
                                        valid = False
                                        break
                                    if d in used:
                                        valid = False
                                        break
                                    new_diffs.add(d)
                                else:
                                    d = row[i] - x
                                    if d <= 0:
                                        valid = False
                                        break
                                    if d in used:
                                        valid = False
                                        break
                                    new_diffs.add(d)
                            if valid:
                                new_val = x
                                break
                        if new_val is not None:
                            break

                    if new_val is not None:
                        row.insert(insert_pos, new_val)
                        for i in range(len(row)):
                            if i < insert_pos:
                                used.add(new_val - row[i])
                            else:
                                used.add(row[i] - new_val)
                        improved = True
                        best_scope = max(max(r) for r in best_rows)
                    else:
                        # Restore
                        row.insert(pos, val)
                        for d in diffs_to_remove:
                            used.add(d)

        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Multiple greedy constructions
    num_greedy = 50
    for i in range(num_greedy):
        if time.time() > deadline - 60:
            break
        rng = random.Random(i * 104729 + 13)
        rows = greedy_build(rng)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Phase 2: Local search on the best solution
    if best_rows is not None:
        remaining_time = deadline - 10
        time_per_ls = max(1.0, remaining_time / 20)
        for i in range(20):
            if time.time() > deadline - 10:
                break
            rng = random.Random(100000 + i * 7919)
            rows_copy = [list(r) for r in best_rows]
            improved_rows, scope = local_search(rows_copy, rng, time.time() + time_per_ls)
            if improved_rows is not None and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in improved_rows]
                if best_scope == lb:
                    break

    # Phase 3: More greedy + local search combinations
    for i in range(30):
        if time.time() > deadline - 15:
            break
        rng = random.Random(200000 + i * 104729)
        rows = greedy_build(rng)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                # Try local search
                remaining = deadline - 15
                if remaining > 2:
                    ls_rows, ls_scope = local_search([list(r) for r in rows], rng, time.time() + min(remaining / 3, 5.0))
                    if ls_rows is not None and ls_scope < best_scope:
                        best_scope = ls_scope
                        best_rows = [list(r) for r in ls_rows]
                else:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    return best_rows
