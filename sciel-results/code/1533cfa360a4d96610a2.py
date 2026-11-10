"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using randomized greedy restarts
    with targeted local search to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

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

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_initial(n_rows, k, rng, perturb=0):
        """
        Greedy initialization with randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if time.time() > deadline - 10:
                    return None
                # Find all valid candidates in a window
                candidates = []
                search_limit = x + 30 + (perturb * 15)
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
                    if time.time() > deadline - 10:
                        return None

                if not candidates:
                    # Expand search
                    x = search_limit
                    continue

                # Choose candidate: smallest, or random among top few
                if len(candidates) > 1 and rng.random() < 0.5:
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

    def local_search_reduce_max(rows, n_rows, k, max_iters=5000):
        """
        Local search specifically designed to reduce the maximum entry.
        Focus on the row with the largest last element and try to reduce it.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        best_rows = [row[:] for row in rows]

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
                        best_rows = [r[:] for r in rows]
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
                                best_rows = [r[:] for r in rows]
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

            # If nothing found, break to avoid infinite loop
            if not found:
                break

        return best_rows, best_score

    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # First, try the deterministic greedy
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            # Try to improve with local search
            improved, improved_score = local_search_reduce_max(det_rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # Then, try randomized restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break

        perturb = rng.randint(0, 5)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            # Try to improve with local search
            improved, improved_score = local_search_reduce_max(rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
# EVOLVE-BLOCK-END
