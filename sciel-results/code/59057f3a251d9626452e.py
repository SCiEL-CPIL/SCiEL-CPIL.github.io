import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local improvement to reduce scope (element shifts)
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Difference-set based search: find difference sets that embed as valid rows
    best_solution = _diffset_search(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Try to find better solutions with randomized backtracking
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(30):
            if time.time() > deadline - 8.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 5:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break

    # 5. Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)

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

def _diffset_search(solution, n_rows, k, deadline):
    """
    Difference-set based search. Each row is characterized by its set of differences.
    We search for difference sets of size k*(k+1)/2 that can be embedded as valid rows,
    minimizing the maximum element (which equals the max difference in the set).
    
    Key insight: a difference set D of size C(k+1,2) embeds as a valid row iff
    there exists a permutation of k+1 elements starting at 0 such that all pairwise
    differences are in D. The max element equals max(D).
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2

    # Collect all used differences
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])
        return used

    used_diffs = compute_used_diffs(best)
    rng = random.Random(123)

    # For each row, try to find a better difference set with smaller max
    for row_idx in range(n_rows):
        if time.time() > deadline - 3.0:
            break

        current_row = best[row_idx]
        current_max = current_row[-1]

        # Try to find a difference set with max < current_max
        # that doesn't conflict with other rows' differences
        other_diffs = set()
        for other_idx in range(n_rows):
            if other_idx == row_idx:
                continue
            other_row = best[other_idx]
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    other_diffs.add(other_row[j] - other_row[i])

        # Search for a valid difference set
        for attempt in range(100):
            if time.time() > deadline - 3.0:
                break

            # Generate a random difference set candidate
            # Start with the current one and perturb
            current_diffs = set()
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    current_diffs.add(current_row[j] - current_row[i])

            # Try to reduce the max by replacing large differences
            max_diff = max(current_diffs)
            if max_diff <= other_diffs or max_diff < 2:
                continue

            # Try replacing the largest difference with a smaller one
            new_diffs = set(current_diffs)
            # Remove the max and try to find a replacement
            candidates = list(range(1, max_diff))
            rng.shuffle(candidates)

            found_replacement = False
            for cand in candidates[:50]:
                if cand in other_diffs:
                    continue
                if cand in new_diffs:
                    continue

                # Tentatively replace
                new_diffs.discard(max_diff)
                new_diffs.add(cand)

                # Check if this difference set can be embedded as a valid row
                row = _embed_diffset(new_diffs, k_plus_1, other_diffs)
                if row is not None:
                    best[row_idx] = row
                    used_diffs = compute_used_diffs(best)
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        found_replacement = True
                        break

            if found_replacement:
                break

    return best

def _embed_diffset(diffset, k_plus_1, forbidden):
    """
    Try to embed a difference set as a valid row of k+1 elements starting at 0.
    Returns the row if successful, None otherwise.
    """
    if len(diffset) != k_plus_1 * (k_plus_1 - 1) // 2:
        return None

    # Backtracking: build the row element by element
    # First element is 0
    row = [0]
    remaining = set(diffset)

    def backtrack(pos):
        if pos == k_plus_1:
            return True

        # Try all candidates for next element
        # The new element x must satisfy: x - row[i] in remaining for all i < pos
        # and x - row[i] not in forbidden
        # Also x - row[i] must not be in the differences already assigned

        # Collect constraints: x must be such that x - row[i] in remaining for all i
        # This means x must be in intersection of (row[i] + remaining) for all i < pos

        # For efficiency, try small values first
        max_possible = max(remaining)
        for x in range(row[-1] + 1, max_possible + 1):
            # Check if all differences x - row[i] are in remaining
            valid = True
            new_diffs = []
            for i in range(pos):
                d = x - row[i]
                if d not in remaining or d in forbidden:
                    valid = False
                    break
                new_diffs.append(d)

            if not valid:
                continue

            # Check that new_diffs are all distinct and not in existing row diffs
            existing_diffs = set()
            for i in range(pos):
                for j in range(i + 1, pos):
                    existing_diffs.add(row[j] - row[i])

            if any(d in existing_diffs for d in new_diffs):
                continue

            # Also check that new_diffs don't conflict with each other
            if len(set(new_diffs)) != len(new_diffs):
                continue

            # Tentatively add x
            row.append(x)
            for d in new_diffs:
                remaining.discard(d)

            if backtrack(pos + 1):
                return True

            # Undo
            row.pop()
            for d in new_diffs:
                remaining.add(d)

        return False

    if backtrack(1):
        return row
    return None

def _gap_exchange_search(solution, n_rows, k, deadline):
    """
    Multi-start local search using gap representation and cross-row gap exchanges.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    def row_to_gaps(row):
        return [row[i+1] - row[i] for i in range(k)]

    def gaps_to_row(gaps):
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        return row

    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])
        return used

    used_diffs = compute_used_diffs(best)
    rng = random.Random(42)

    improved = True
    iteration = 0
    while improved and time.time() < deadline - 3.0:
        improved = False
        iteration += 1

        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                for new_val in range(row[pos-1] + 1, original):
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

                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    if any(d <= 0 for d in new_diffs):
                        continue

                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
                if improved:
                    break

        if time.time() < deadline - 3.0:
            for attempt in range(5):
                if time.time() > deadline - 3.0:
                    break

                r1, r2 = rng.sample(range(n_rows), 2)
                pos = rng.randint(0, k - 1)

                row1 = best[r1]
                row2 = best[r2]

                gap1 = row1[pos+1] - row1[pos]
                gap2 = row2[pos+1] - row2[pos]

                if gap1 == gap2:
                    continue

                new_row1 = list(row1)
                new_row2 = list(row2)

                delta1 = gap2 - gap1
                for i in range(pos+1, k_plus_1):
                    new_row1[i] = row1[i] + delta1

                if new_row1[0] != 0 or any(new_row1[i] >= new_row1[i+1] for i in range(k)):
                    continue
                if new_row1[-1] > best_scope:
                    continue

                delta2 = gap1 - gap2
                for i in range(pos+1, k_plus_1):
                    new_row2[i] = row2[i] + delta2

                if new_row2[0] != 0 or any(new_row2[i] >= new_row2[i+1] for i in range(k)):
                    continue
                if new_row2[-1] > best_scope:
                    continue

                new_diffs1 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        new_diffs1.add(new_row1[j] - new_row1[i])

                new_diffs2 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        new_diffs2.add(new_row2[j] - new_row2[i])

                old_diffs1 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        old_diffs1.add(row1[j] - row1[i])

                old_diffs2 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        old_diffs2.add(row2[j] - row2[i])

                other_diffs = used_diffs - old_diffs1 - old_diffs2
                if (new_diffs1 | new_diffs2) & other_diffs:
                    continue
                if new_diffs1 & new_diffs2:
                    continue

                used_diffs.difference_update(old_diffs1)
                used_diffs.difference_update(old_diffs2)
                used_diffs.update(new_diffs1)
                used_diffs.update(new_diffs2)
                best[r1] = new_row1
                best[r2] = new_row2
                improved = True
                break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with MRV heuristic.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)

    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        valid_candidates = []
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        n_smallest = min(12, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]

        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest

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
                used_diffs[row[i2] - row[j]] = True

    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope by shifting elements down.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    improved = True
    while improved and time.time() < deadline - 2.0:
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

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _polish_scope(solution, n_rows, k, deadline):
    """
    Final polish: try to reduce scope by 1 at a time.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)

    while best_scope > 0 and time.time() < deadline - 2.0:
        target_scope = best_scope - 1
        if target_scope < 0:
            break

        found = False
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break

            sol = _backtracking_search(n_rows, k, target_scope, deadline, attempt * 7 + 13)
            if sol:
                new_scope = max(max(row) for row in sol)
                if new_scope <= target_scope:
                    best = sol
                    best_scope = new_scope
                    found = True
                    break

        if not found:
            break

    return best
