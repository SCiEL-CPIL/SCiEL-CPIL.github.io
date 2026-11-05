import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit

    best_rows = None
    best_score = float('inf')

    # Precompute expected number of differences per row
    expected_diffs = (k + 1) * k // 2

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == expected_diffs

    def get_valid_next(row, used, cap):
        valids = []
        last = row[-1]
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(row):
                valids.append(x)
        return valids

    def randomized_greedy(max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score)

                valids = get_valid_next(row, used, cap)
                if not valids:
                    return None

                # Weighted random choice, favoring smaller x
                if len(valids) > 1:
                    consider = valids[:30]
                    weights = [1.0 / (v + 1) for v in consider]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = consider[0]
                    for v, w in zip(consider, weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = valids[0]

                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    def local_search(rows):
        """Optimized local search: precompute other_diffs once per row, try to reduce each element."""
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Shuffle row order to avoid bias
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            # Try to improve each row
            for i in row_order:
                if time.time() - start_time >= time_limit:
                    break
                row = rows[i]

                # Precompute diffs of all OTHER rows ONCE for this row
                other_diffs = set()
                for r_idx, r in enumerate(rows):
                    if r_idx != i:
                        other_diffs.update(row_diffs(r))

                # Iterate from last element to first (excluding 0)
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start_time >= time_limit:
                        break
                    current_val = row[pos]
                    # Try smaller values
                    min_val = row[pos - 1] + 1
                    max_val = current_val - 1  # We want to reduce

                    if max_val < min_val:
                        continue

                    # Try values from smallest to largest
                    found = False
                    for v in range(min_val, max_val + 1):
                        # Create a copy of the row with the change
                        candidate_row = list(row)
                        candidate_row[pos] = v

                        # Check strictly increasing
                        is_valid = True
                        for j in range(1, len(candidate_row)):
                            if candidate_row[j] <= candidate_row[j - 1]:
                                is_valid = False
                                break

                        if not is_valid:
                            continue

                        # Get diffs of candidate row
                        cand_diffs = row_diffs(candidate_row)

                        # Check for internal duplicates
                        if len(cand_diffs) != expected_diffs:
                            continue

                        # Check for global duplicates with other rows
                        if cand_diffs & other_diffs:
                            continue

                        # Accept the change
                        rows[i] = candidate_row
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        found = True
                        break  # Break inner for v loop

                    if found:
                        # Recompute other_diffs since row i changed
                        other_diffs = set()
                        for r_idx, r in enumerate(rows):
                            if r_idx != i:
                                other_diffs.update(row_diffs(r))
                        break  # Break inner for pos loop
                if improved:
                    break  # Break outer for i loop
        return rows, current_max

    def swap_local_search(rows):
        """Try swapping two rows or reordering elements within a row to reduce scope."""
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try swapping pairs of rows
            for i in range(n_rows):
                if time.time() - start_time >= time_limit:
                    break
                for j in range(i + 1, n_rows):
                    if time.time() - start_time >= time_limit:
                        break
                    # Swap rows i and j
                    rows[i], rows[j] = rows[j], rows[i]
                    new_max = max(max(r) for r in rows)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    else:
                        # Swap back
                        rows[i], rows[j] = rows[j], rows[i]
        return rows, current_max

    def polish_pass(rows):
        """Final high-precision polish: multiple random orderings of local search."""
        current_max = max(max(r) for r in rows)
        for _ in range(5):
            if time.time() - start_time >= time_limit - 3:
                break
            shuffled = [list(r) for r in rows]
            improved_rows, improved_score = local_search(shuffled)
            if validate_solution(improved_rows):
                if improved_score < current_max:
                    current_max = improved_score
                    rows = improved_rows
        return rows, current_max

    def validate_solution(rows):
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            d_set = row_diffs(r)
            if all_diffs & d_set:
                return False
            all_diffs.update(d_set)
        return True

    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        # Try randomized greedy
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 5:
                    improved_rows, improved_score = local_search(best_rows)
                    # Validate improved_rows before accepting
                    if validate_solution(improved_rows):
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = improved_rows
                    # Try swap local search
                    if time.time() - start_time < time_limit - 5:
                        swapped_rows, swapped_score = swap_local_search(best_rows)
                        if validate_solution(swapped_rows):
                            if swapped_score < best_score:
                                best_score = swapped_score
                                best_rows = swapped_rows
                    # Final polish pass
                    if time.time() - start_time < time_limit - 3:
                        polished_rows, polished_score = polish_pass(best_rows)
                        if validate_solution(polished_rows):
                            if polished_score < best_score:
                                best_score = polished_score
                                best_rows = polished_rows
                if best_score <= 105:
                    break

        attempts += 1
        if attempts % 50 == 0:
            # Occasionally try a more aggressive local search by shuffling row order
            if best_rows is not None and time.time() - start_time < time_limit - 10:
                shuffled = list(best_rows)
                random.shuffle(shuffled)
                improved_rows, improved_score = local_search(shuffled)
                if validate_solution(improved_rows):
                    if improved_score < best_score:
                        best_score = improved_score
                        best_rows = improved_rows
                # Also try swap search on shuffled
                if time.time() - start_time < time_limit - 10:
                    swapped_rows, swapped_score = swap_local_search(shuffled)
                    if validate_solution(swapped_rows):
                        if swapped_score < best_score:
                            best_score = swapped_score
                            best_rows = swapped_rows
                # Polish
                if time.time() - start_time < time_limit - 5:
                    polished_rows, polished_score = polish_pass(best_rows)
                    if validate_solution(polished_rows):
                        if polished_score < best_score:
                            best_score = polished_score
                            best_rows = polished_rows

    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
