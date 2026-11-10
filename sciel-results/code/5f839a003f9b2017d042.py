import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using simulated annealing with
    multiple move types to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry

    def greedy_initial(n_rows, k, rng=None, perturb=0):
        """
        Greedy initialization with optional randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                candidates = []
                search_limit = x + 50 + (perturb * 10)
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 20:
                        return None

                if not candidates:
                    x = search_limit
                    continue

                if rng is not None and len(candidates) > 1:
                    top = min(3, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def get_row_diffs(row, ci):
        """Get all differences involving position ci in row."""
        diffs = set()
        for i in range(ci):
            diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            diffs.add(row[j] - row[ci])
        return diffs

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]

        old_diffs = get_row_diffs(row, ci)
        temp_used = used_diffs - old_diffs

        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None

        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)

        if not valid:
            return False, None

        new_used = temp_used | new_diffs
        return True, new_used

    def try_swap_elements(rows, ri1, ci1, ri2, ci2, used_diffs):
        """
        Try to swap elements (ri1, ci1) and (ri2, ci2).
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row1 = rows[ri1]
        row2 = rows[ri2]
        val1 = row1[ci1]
        val2 = row2[ci2]

        if ri1 == ri2:
            # Same row swap - just check ordering
            if ci1 < ci2 and val2 <= row1[ci1 - 1] if ci1 > 0 else False:
                return False, None
            if ci1 < ci2 and val2 >= row1[ci1 + 1] if ci1 < len(row1) - 1 else False:
                return False, None
            if ci1 > ci2 and val1 <= row2[ci2 - 1] if ci2 > 0 else False:
                return False, None
            if ci1 > ci2 and val1 >= row2[ci2 + 1] if ci2 < len(row2) - 1 else False:
                return False, None

        # Collect all old differences
        old_diffs1 = get_row_diffs(row1, ci1)
        old_diffs2 = get_row_diffs(row2, ci2)
        all_old_diffs = old_diffs1 | old_diffs2

        temp_used = used_diffs - all_old_diffs

        # Check if swap is valid
        valid = True

        # Check val2 in row1 at position ci1
        if ri1 != ri2:
            if ci1 > 0 and val2 <= row1[ci1 - 1]:
                valid = False
            if ci1 < len(row1) - 1 and val2 >= row1[ci1 + 1]:
                valid = False

        if valid:
            for i in range(ci1):
                d = val2 - row1[i]
                if d <= 0 or d in temp_used:
                    valid = False
                    break
            if valid:
                for j in range(ci1 + 1, len(row1)):
                    d = row1[j] - val2
                    if d <= 0 or d in temp_used:
                        valid = False
                        break

        # Check val1 in row2 at position ci2
        if valid and ri1 != ri2:
            if ci2 > 0 and val1 <= row2[ci2 - 1]:
                valid = False
            if ci2 < len(row2) - 1 and val1 >= row2[ci2 + 1]:
                valid = False

        if valid and ri1 != ri2:
            for i in range(ci2):
                d = val1 - row2[i]
                if d <= 0 or d in temp_used:
                    valid = False
                    break
            if valid:
                for j in range(ci2 + 1, len(row2)):
                    d = row2[j] - val1
                    if d <= 0 or d in temp_used:
                        valid = False
                        break

        if not valid:
            return False, None

        # Success
        rows[ri1][ci1] = val2
        rows[ri2][ci2] = val1

        new_diffs1 = get_row_diffs(rows[ri1], ci1)
        new_diffs2 = get_row_diffs(rows[ri2], ci2)
        new_used = temp_used | new_diffs1 | new_diffs2
        return True, new_used

    def simulated_annealing(rows, n_rows, k, max_iters=5000):
        """
        Simulated annealing to minimize the largest entry.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_rows = [row[:] for row in rows]
        best_score = max(row[-1] for row in rows)

        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.9995

        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Choose a random move
            move_type = random.random()

            if move_type < 0.6:
                # Move a single element
                ri = random.randint(0, n_rows - 1)
                ci = random.randint(1, len(rows[ri]) - 1)
                current = rows[ri][ci]

                # Try to move to a smaller value
                if ci > 1:
                    min_val = rows[ri][ci - 1] + 1
                else:
                    min_val = 1
                max_val = rows[ri][ci] - 1 if ci < len(rows[ri]) - 1 else current + 10

                if max_val >= min_val:
                    new_val = random.randint(min_val, max_val)
                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if success:
                        new_score = max(r[-1] for r in rows)
                        delta = new_score - best_score
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            used_diffs = new_used
                            if new_score < best_score:
                                best_score = new_score
                                best_rows = [row[:] for row in rows]
                        else:
                            # Revert
                            rows[ri][ci] = current
                else:
                    # Try larger value
                    new_val = current + random.randint(1, 5)
                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if success:
                        new_score = max(r[-1] for r in rows)
                        delta = new_score - best_score
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            used_diffs = new_used
                            if new_score < best_score:
                                best_score = new_score
                                best_rows = [row[:] for row in rows]
                        else:
                            rows[ri][ci] = current

            elif move_type < 0.8:
                # Swap two elements from different rows
                ri1 = random.randint(0, n_rows - 1)
                ri2 = random.randint(0, n_rows - 1)
                if ri1 == ri2:
                    continue
                ci1 = random.randint(1, len(rows[ri1]) - 1)
                ci2 = random.randint(1, len(rows[ri2]) - 1)

                success, new_used = try_swap_elements(rows, ri1, ci1, ri2, ci2, used_diffs)
                if success:
                    new_score = max(r[-1] for r in rows)
                    delta = new_score - best_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        used_diffs = new_used
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [row[:] for row in rows]
                    else:
                        # Revert swap
                        temp = rows[ri1][ci1]
                        rows[ri1][ci1] = rows[ri2][ci2]
                        rows[ri2][ci2] = temp

            else:
                # Shift an element by a small amount
                ri = random.randint(0, n_rows - 1)
                ci = random.randint(1, len(rows[ri]) - 1)
                current = rows[ri][ci]
                shift = random.choice([-2, -1, 1, 2])
                new_val = current + shift

                if ci > 0 and new_val <= rows[ri][ci - 1]:
                    continue
                if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
                    continue

                success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                if success:
                    new_score = max(r[-1] for r in rows)
                    delta = new_score - best_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        used_diffs = new_used
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [row[:] for row in rows]
                    else:
                        rows[ri][ci] = current

            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_rows, best_score

    def local_search_reduce_max(rows, n_rows, k, max_iters=2000):
        """
        Local search specifically designed to reduce the maximum entry.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)

        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break

            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            max_val = rows[max_row_idx][-1]
            row = rows[max_row_idx]
            ci = len(row) - 1

            found = False
            for new_val in range(max_val - 1, row[ci - 1], -1):
                success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                if success:
                    rows[max_row_idx][ci] = new_val
                    used_diffs = new_used
                    new_score = max(r[-1] for r in rows)
                    if new_score < best_score:
                        best_score = new_score
                    found = True
                    break

            if not found:
                for ci in range(len(row) - 2, 0, -1):
                    current = row[ci]
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                        if success:
                            rows[max_row_idx][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            found = True
                            break
                    if found:
                        break

            if not found:
                for ri in range(n_rows):
                    if ri == max_row_idx:
                        continue
                    row = rows[ri]
                    for ci in range(len(row) - 1, 0, -1):
                        current = row[ci]
                        for new_val in range(current - 1, row[ci - 1], -1):
                            success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                            if success:
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            if not found:
                break

        return rows, best_score

    # Phase 1: Multiple randomized greedy restarts
    best_rows = None
    best_score = float('inf')
    rng = random.Random(42)

    # First, deterministic greedy as baseline
    det_rows = greedy_initial(n_rows, k, rng=None)
    if det_rows is not None:
        valid, score = validate(det_rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in det_rows]

    # Randomized greedy restarts
    n_restarts = 30
    for restart in range(n_restarts):
        if time_remaining() < 30:
            break

        perturb = restart % 5
        seed_val = 42 + restart * 7
        rng_restart = random.Random(seed_val)

        rows = greedy_initial(n_rows, k, rng=rng_restart, perturb=perturb)
        if rows is None:
            continue

        valid, score = validate(rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]

    if best_rows is None:
        # Fallback
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows

    # Phase 2: Local search on the best solution
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=2000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    # Phase 3: Simulated annealing
    if time_remaining() > 20:
        sa_rows, sa_score = simulated_annealing(best_rows, n_rows, k, max_iters=8000)
        if sa_score < best_score:
            best_score = sa_score
            best_rows = sa_rows

    # Phase 4: More SA restarts
    for restart in range(10):
        if time_remaining() < 15:
            break

        # Perturb the best solution
        perturbed = [row[:] for row in best_rows]
        for ri in range(n_rows):
            for ci in range(1, len(perturbed[ri])):
                if rng.random() < 0.1:
                    perturbed[ri][ci] += rng.randint(1, 5)

        valid, score = validate(perturbed)
        if not valid:
            continue

        sa_rows, sa_score = simulated_annealing(perturbed, n_rows, k, max_iters=3000)
        if sa_score < best_score:
            best_score = sa_score
            best_rows = sa_rows

    # Final local search
    if time_remaining() > 5:
        improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=1000)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

    return best_rows
# EVOLVE-BLOCK-END
