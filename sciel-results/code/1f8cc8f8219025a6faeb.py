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


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with stronger pruning and randomized candidate ordering.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)
    used_count = 0

    def build_row(pos, current_mark, row_so_far, depth):
        nonlocal used_count
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Prune upper bound
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        # Find valid candidates with early termination
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            # Limit candidate list for speed
            if len(valid_candidates) >= 50:
                break

        if not valid_candidates:
            return None

        # Bias towards smaller values, with some randomness
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        # Occasionally add larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            used_count += len(new_diffs)

            res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
            if res is not None:
                return res

            # Unmark differences
            for d in new_diffs:
                used_diffs[d] = False
            used_count -= len(new_diffs)

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                if not used_diffs[d]:
                    used_diffs[d] = True
                    used_count += 1

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: repeatedly pull each element to its smallest
    feasible value, then try 2-element swaps to escape local minima.
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

    def _pull_to_smallest(row_idx):
        """Pull all elements in row row_idx to their smallest feasible values."""
        row = best[row_idx]
        any_changed = False
        # Process from left to right (after pos 0)
        for pos in range(1, k_plus_1):
            # Find smallest feasible value > row[pos-1]
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                # Check conflicts
                new_diffs = set()
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        break
                    new_diffs.add(d)
                else:
                    # Check diffs with elements after pos
                    ok = True
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - row[pos])
                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_changed = True
                    break
        return any_changed

    # Phase 1: Repeatedly pull all rows to smallest feasible
    for iteration in range(30):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Try 2-element swaps to escape local minima
    for iteration in range(15):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos1 in range(1, k_plus_1):
                for pos2 in range(pos1 + 1, k_plus_1):
                    # Try swapping row[pos1] and row[pos2]
                    v1, v2 = row[pos1], row[pos2]
                    if v1 == v2:
                        continue
                    # After swap, check increasing order
                    if pos1 > 0 and v2 <= row[pos1 - 1]:
                        continue
                    if pos2 < k_plus_1 - 1 and v1 >= row[pos2 + 1]:
                        continue
                    if v1 >= v2:
                        continue
                    # Check feasibility
                    new_diffs = set()
                    for i in range(pos1):
                        d = v2 - row[i]
                        if d <= 0:
                            break
                        new_diffs.add(d)
                    else:
                        ok = True
                        for i in range(pos1 + 1, pos2):
                            d = v2 - row[i]
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                        if ok:
                            for i in range(pos2 + 1, k_plus_1):
                                d = row[i] - v2
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            for i in range(pos1 + 1, pos2):
                                d = v1 - row[i]
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            for i in range(pos2 + 1, k_plus_1):
                                d = row[i] - v1
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            # Remove old diffs
                            old_diffs = set()
                            for i in range(pos1):
                                old_diffs.add(v1 - row[i])
                            for i in range(pos1 + 1, pos2):
                                old_diffs.add(v1 - row[i])
                            for i in range(pos2 + 1, k_plus_1):
                                old_diffs.add(row[i] - v1)
                            for i in range(pos1):
                                old_diffs.add(v2 - row[i])
                            for i in range(pos1 + 1, pos2):
                                old_diffs.add(v2 - row[i])
                            for i in range(pos2 + 1, k_plus_1):
                                old_diffs.add(row[i] - v2)
                            old_diffs.add(v2 - v1)
                            # Check no conflict
                            if not (new_diffs & (used_diffs - old_diffs)):
                                used_diffs.difference_update(old_diffs)
                                used_diffs.update(new_diffs)
                                row[pos1], row[pos2] = v2, v1
                                any_changed = True
                                break
                if any_changed:
                    break
            if any_changed:
                break
        if not any_changed:
            break

    # Phase 3: Final pull pass
    for iteration in range(10):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    High-precision polish: pull-to-smallest with randomized restarts and
    2-element perturbations to escape local minima.
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

    def _pull_all():
        """Pull all rows to smallest feasible values."""
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(1, k_plus_1):
                for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        for j in range(pos + 1, k_plus_1):
                            d = row[j] - new_val
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(row[pos] - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - row[pos])
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        any_changed = True
                        break
        return any_changed

    # Multiple rounds of pull + perturb
    for round_num in range(25):
        if time.time() > deadline - 3.0:
            break

        # Pull to smallest
        _pull_all()

        # Random perturbation: pick a random element and try a few nearby values
        if round_num % 3 == 2:  # every 3rd round
            row_idx = round_num % n_rows
            row = best[row_idx]
            for _ in range(5):
                pos = 1 + (round_num * 7 + _ * 13) % (k_plus_1 - 1)
                original = row[pos]
                # Try a few values above original (perturbation)
                for offset in [1, 2, 3, 5, 8]:
                    new_val = original + offset
                    if new_val > 200:  # safety
                        break
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        for j in range(pos + 1, k_plus_1):
                            d = row[j] - new_val
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(original - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - original)
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        break

        # Pull again after perturbation
        _pull_all()

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
