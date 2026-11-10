import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using randomized greedy restarts
    with targeted local search to minimize the largest entry.
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
        If rng is provided, we randomize the choice among valid candidates
        and add small perturbations to explore different solution paths.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                # Find all valid candidates in a window
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
                    # Expand search
                    x = search_limit
                    continue

                # Choose candidate: smallest, or random among top few
                if rng is not None and len(candidates) > 1:
                    # Pick randomly from top 3 for diversity
                    top = min(3, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)

                # Skip ahead to avoid re-checking
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

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]

        # Collect old differences involving position ci
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])

        # Remove old differences
        temp_used = used_diffs - old_diffs

        # Check if new_val is valid
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

        # Success: return the new used set
        new_used = temp_used | new_diffs
        return True, new_used

    def local_search_reduce_max(rows, n_rows, k, max_iters=2000):
        """
        Local search specifically designed to reduce the maximum entry.
        Focus on the row with the largest last element and try to reduce it.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)

        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            max_val = rows[max_row_idx][-1]

            # Try to reduce the last element of the max row
            row = rows[max_row_idx]
            ci = len(row) - 1  # Last element

            # Try smaller values for the last element
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

            # If we couldn't reduce the last element, try other positions in that row
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

            # Also try to reduce elements in other rows to free up differences
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

    def local_search_general(rows, n_rows, k, max_iters=1500):
        """
        General local search: try to reduce any element to a smaller valid position.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)

        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break

            made_progress = False

            # Try to reduce elements in all rows
            for ri in range(n_rows):
                row = rows[ri]
                for ci in range(len(row) - 1, 0, -1):
                    current = row[ci]
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            rows[ri][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            made_progress = True
                            break
                    if made_progress:
                        break
                if made_progress:
                    break

            if not made_progress:
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

    # Randomized greedy restarts - the key improvement
    n_restarts = 50
    for restart in range(n_restarts):
        if time_remaining() < 25:
            break

        # Vary the randomization parameters
        perturb = restart % 5  # Different perturbation levels
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
        # Fallback: simple construction
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
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=3000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    improved_rows, improved_score = local_search_general(best_rows, n_rows, k, max_iters=2000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    # Phase 3: More randomized restarts with perturbation of best solution
    for restart in range(30):
        if time_remaining() < 20:
            break

        # Perturb the best solution
        perturbed = [row[:] for row in best_rows]
        for ri in range(n_rows):
            for ci in range(1, len(perturbed[ri])):
                if rng.random() < 0.15:
                    perturbed[ri][ci] += rng.randint(1, 10)

        valid, score = validate(perturbed)
        if not valid:
            continue

        improved_rows, improved_score = local_search_reduce_max(perturbed, n_rows, k, max_iters=1500)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

        improved_rows, improved_score = local_search_general(best_rows, n_rows, k, max_iters=1000)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

    return best_rows
# EVOLVE-BLOCK-END
