import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 118  # Tighter safety margin

    # Generate initial solution using fast greedy construction
    best_solution = _greedy_construct_fast(n_rows, k)
    if not best_solution:
        best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Phase 1: Aggressive local search to compress
    improved = _local_search_minimize(best_solution, n_rows, k, deadline - 80)
    if improved:
        curr_score = max(max(r) for r in improved)
        if curr_score < best_score:
            best_score = curr_score
            best_solution = improved

    # Phase 2: Iterated Local Search with perturbations
    no_improve_count = 0
    while time.time() < deadline - 25:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Perform several random valid moves (perturbation)
        n_moves = random.randint(2, 6)
        for _ in range(n_moves):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't touch position 0

            # Find a random valid new value for candidate[r_idx][pos]
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline - 20)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # If we've had many iterations without improvement, try a stronger perturbation
        if no_improve_count >= 4:
            no_improve_count = 0
            # Try a more aggressive perturbation: reset one row
            if n_rows > 1:
                r_idx = random.randint(0, n_rows - 1)
                # Rebuild this row greedily given the other rows
                other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                used_diffs_other = _compute_all_diffs(other_rows)
                new_row = _greedy_row_given_diffs(used_diffs_other, k)
                if new_row is not None:
                    candidate[r_idx] = new_row
                    improved2 = _local_search_minimize(candidate, n_rows, k, deadline - 20)
                    if improved2:
                        curr_score2 = max(max(r) for r in improved2)
                        if curr_score2 < best_score:
                            best_score = curr_score2
                            best_solution = improved2

    # Phase 3: Final high-precision polish with remaining time
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline - 1)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution


def _greedy_row_given_diffs(used_diffs, k):
    """Build a single row greedily given already-used differences."""
    row = [0]
    current_max = 0
    while len(row) < k + 1:
        found = False
        for x in range(current_max + 1, current_max + 2000):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break

            if valid:
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                current_max = x
                found = True
                break
        if not found:
            return None
    return row

def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _compute_all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]

    # Compute all diffs once
    all_diffs = _compute_all_diffs(rows)

    # Diffs involving the old value at this position
    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    current_max = max(max(r) for r in rows)
    upper_bound = min(current_max + 30, 250)
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates, biasing toward lower values
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val

            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)

        if valid:
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Weighted random choice favoring smaller values
    if len(valid_candidates) == 1:
        return valid_candidates[0]

    max_candidate = max(valid_candidates)
    total_weight = 0
    weights = []
    for v in valid_candidates:
        weight = (max_candidate - v + 1) ** 2
        weights.append(weight)
        total_weight += weight

    r = random.uniform(0, total_weight)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return valid_candidates[i]
    return valid_candidates[-1]

def _local_search_minimize(rows, n_rows, k, deadline):
    """Multi-pass local search to minimize max entry.

    Pass 1: Right-to-left greedy lowering per row.
    Pass 2: Left-to-right re-check to catch newly enabled reductions.
    Repeats until no progress or time runs out.
    """
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    while time.time() < deadline - 2:
        progress = False

        # Pass 1: Right-to-left
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 2:
                break

            row = best_rows[r_idx]

            for pos in range(k, 0, -1):
                old_val = row[pos]

                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        found = new_val
                        break

                if found is not None and found < old_val:
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        # Pass 2: Left-to-right to catch cascading improvements
        if time.time() < deadline - 2:
            for r_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break

                row = best_rows[r_idx]

                for pos in range(1, k + 1):
                    old_val = row[pos]

                    diffs_involving_old = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)

                    forbidden_diffs = used_diffs - diffs_involving_old

                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < len(row):
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if upper_bound < lower_bound:
                        continue

                    found = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val

                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)

                        if valid:
                            found = new_val
                            break

                    if found is not None and found < old_val:
                        for d in diffs_involving_old:
                            used_diffs.discard(d)
                        for d in new_diffs:
                            used_diffs.add(d)
                        row[pos] = found
                        progress = True

        if not progress:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """Greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 2000):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break

                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows


def _greedy_construct_fast(n_rows, k):
    """Faster greedy: try multiple random starting offsets to find compact rows."""
    best_rows = None
    best_max = float('inf')

    for attempt in range(20):
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                # Try a small random window first, then expand
                search_limit = current_max + 500
                for x in range(current_max + 1, search_limit):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break

                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        current_max = x
                        found = True
                        break

                if not found:
                    # Expand search
                    for x in range(search_limit, current_max + 3000):
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break

                        if valid:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            current_max = x
                            found = True
                            break

                if not found:
                    success = False
                    break

            if not success:
                break

            rows.append(row)

        if success and rows:
            curr_max = max(max(r) for r in rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = rows

    if best_rows:
        return best_rows
    return _greedy_construct(n_rows, k)
