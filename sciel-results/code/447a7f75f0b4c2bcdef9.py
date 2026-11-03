import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1

    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass for further compaction
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Try to find better solutions with randomized backtracking
    # for target scopes below current best
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        # Try multiple random restarts for this target scope
        found = False
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    # Polish
                    polished = _polish_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in polished)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 5:
            # If we can't find anything near the lower bound, stop
            if target_scope < lower_bound:
                break

    # Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    return best_solution


def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _spectral_greedy(n_rows, k, deadline):
    """
    Spectral relaxation-guided greedy construction.
    Uses a simple priority-based greedy to construct rows.
    """
    k_plus_1 = k + 1
    max_scope = 200  # Initial estimate

    # Construct rows greedily with priority on smaller differences
    used = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k_plus_1:
            if time.time() > deadline:
                return None
            x += 1
            if x > max_scope:
                return None
            new_diffs = {x - a for a in row}
            if any(d in used for d in new_diffs):
                continue
            # Check if all new diffs are within scope
            if max(new_diffs) > max_scope:
                continue
            # Accept this x
            row.append(x)
            used.update(new_diffs)

        rows.append(row)

    # Verify solution
    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)

    return rows


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses bitmask for fast difference checking with MRV heuristic.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    # Use a bitmask for used differences
    # Bit d is set if difference d is already used
    used_mask = 0  # Python int, arbitrary precision

    def count_valid_candidates(row_so_far, current_mark, mask, max_x):
        """Count how many valid candidates exist for the next position."""
        count = 0
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    conflict = True
                    break
                if (mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                count += 1
        return count

    def build_row_mrv(pos, current_mark, row_so_far, mask):
        """
        Build a row using MRV (Minimum Remaining Values) heuristic.
        At each step, choose the position with the fewest valid candidates.
        """
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Find valid candidates for current position
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    conflict = True
                    break
                if (mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        # MRV: For the next positions, estimate how many candidates would be available
        # This helps us choose values that leave more flexibility for future positions
        # We prioritize candidates that leave the most options for subsequent positions

        # Simple heuristic: try smallest candidates first, but with some randomization
        # to escape local minima
        n_to_consider = min(20, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_to_consider]

        # Add some random larger candidates for diversity
        if len(valid_candidates) > n_to_consider and rng.random() < 0.3:
            extra_pool = valid_candidates[n_to_consider:]
            n_extra = min(5, len(extra_pool))
            extra = rng.sample(extra_pool, n_extra)
            candidates_to_try.extend(extra)

        # Sort by a heuristic: prefer smaller values but with some randomness
        rng.shuffle(candidates_to_try)
        candidates_to_try.sort(key=lambda x: x + rng.random() * 10)

        for x in candidates_to_try:
            new_mask = mask
            for a in row_so_far:
                d = x - a
                new_mask |= (1 << d)

            res = build_row_mrv(pos + 1, x, row_so_far + [x], new_mask)
            if res is not None:
                return res

        return None

    for i in range(n_rows):
        row = build_row_mrv(1, 0, [0], used_mask)
        if row is None:
            return None
        rows.append(row)
        # Update mask with all differences in this row
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                used_mask |= (1 << d)

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    Uses a "pull to minimum feasible" strategy: for each element, try to move it
    to the smallest possible value that doesn't create conflicts.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Iteratively try to reduce each element to its minimum feasible value
    # Process from last element to first, and from last row to first
    for iteration in range(50):
        if time.time() > deadline - 2.0:
            break

        any_improved = False

        for row_idx in range(n_rows - 1, -1, -1):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]

                # Try to find the smallest feasible value for this position
                min_feasible = original
                for new_val in range(row[pos - 1] + 1, original):
                    # Check if new value creates conflicts
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # This is feasible, update min_feasible
                    min_feasible = new_val

                # If we found a smaller feasible value, apply it
                if min_feasible < original:
                    new_val = min_feasible
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_improved = True

        if not any_improved:
            break

    return best


def _polish_scope(solution, n_rows, k, deadline):
    """
    Aggressive polishing: repeatedly try single-element shifts to minimize scope.
    Uses a two-phase approach: first pull each element to its minimum feasible,
    then try coordinated moves of pairs of elements.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def try_shift_element(row_idx, pos, new_val, row, used_diffs):
        """Try to shift element at row_idx, pos to new_val. Returns True if successful."""
        original = row[pos]
        if new_val == original:
            return False

        # Check if new value creates conflicts
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, k_plus_1):
            new_diffs.add(row[j] - new_val)

        # Remove old diffs
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(original - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - original)

        # Check if new_diffs conflict with (used_diffs - old_diffs)
        if new_diffs & (used_diffs - old_diffs):
            return False

        # Check if new_diffs are all positive
        if any(d <= 0 for d in new_diffs):
            return False

        # Apply the change
        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        row[pos] = new_val
        return True

    # Phase 1: Pull each element to its minimum feasible value
    for iteration in range(30):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        for row_idx in range(n_rows - 1, -1, -1):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try to find the smallest feasible value
                for new_val in range(row[pos - 1] + 1, original):
                    if try_shift_element(row_idx, pos, new_val, row, used_diffs):
                        any_improved = True
                        break

        if not any_improved:
            break

    # Phase 2: Try coordinated moves of adjacent elements
    for iteration in range(10):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(1, k_plus_1 - 1):
                # Try to shift both pos and pos+1 down together
                orig_pos = row[pos]
                orig_next = row[pos + 1]

                for delta1 in range(1, min(5, orig_pos - row[pos - 1])):
                    new_pos = orig_pos - delta1
                    for delta2 in range(1, min(5, orig_next - new_pos)):
                        new_next = orig_next - delta2

                        # Check if both shifts are feasible simultaneously
                        # Temporarily remove old diffs
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(orig_pos - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - orig_pos)
                        for i in range(pos + 1):
                            old_diffs.add(orig_next - row[i])
                        for j in range(pos + 2, k_plus_1):
                            old_diffs.add(row[j] - orig_next)

                        # Compute new diffs
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_pos - row[i])
                        for j in range(pos + 1, k_plus_1):
                            if j == pos + 1:
                                new_diffs.add(new_next - new_pos)
                            else:
                                new_diffs.add(row[j] - new_pos)
                        for i in range(pos + 1):
                            if i == pos:
                                new_diffs.add(new_next - new_pos)
                            else:
                                new_diffs.add(new_next - row[i])
                        for j in range(pos + 2, k_plus_1):
                            new_diffs.add(row[j] - new_next)

                        # Check conflicts
                        if new_diffs & (used_diffs - old_diffs):
                            continue
                        if any(d <= 0 for d in new_diffs):
                            continue

                        # Apply both changes
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_pos
                        row[pos + 1] = new_next
                        any_improved = True
                        break
                    if any_improved:
                        break
                if any_improved:
                    break

        if not any_improved:
            break

    return best
