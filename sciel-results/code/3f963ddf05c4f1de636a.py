import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Generate initial solution using greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Iterated Local Search with valid perturbations
    while time.time() < deadline - 15:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Perform several random valid moves (perturbation)
        num_moves = random.randint(3, 8)
        for _ in range(num_moves):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't touch position 0

            # Find a random valid new value for candidate[r_idx][pos]
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val

        # Occasionally do a more aggressive perturbation: swap two positions in a row
        if random.random() < 0.3:
            r_idx = random.randint(0, n_rows - 1)
            pos1 = random.randint(1, k - 1)
            pos2 = random.randint(pos1 + 1, k)
            # Try swapping values (must maintain increasing order, so only if they can be reordered)
            # Instead, try to lower the higher position
            row = candidate[r_idx]
            old_val = row[pos2]
            new_val = _find_random_valid_value(candidate, r_idx, pos2, n_rows, k)
            if new_val is not None and new_val < old_val:
                candidate[r_idx][pos2] = new_val

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved

    # Final polish pass with remaining time
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution

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
    row_len = k + 1

    # Build the set of differences not involving the old value at this position
    all_diffs = _compute_all_diffs(rows)

    diffs_involving_old = set()
    for i in range(row_len):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value - allow some expansion beyond current max
    lower_bound = row[pos - 1] + 1
    current_max = max(max(r) for r in rows)
    upper_bound = min(current_max + 20, 350)  # Allow slight expansion
    if pos + 1 < row_len:
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        # Check validity: each new diff must not collide with forbidden_diffs
        # or with another new diff in the same row.
        valid = True
        new_diffs = set()
        for i in range(row_len):
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

    # Pick randomly, preferring values close to current
    if len(valid_candidates) > 5:
        # Weight toward values near the current position
        idx = min(len(valid_candidates) - 1, max(0, valid_candidates.index(old_val) + random.randint(-2, 2)) if old_val in valid_candidates else random.randint(0, len(valid_candidates) - 1))
        return valid_candidates[idx]
    return random.choice(valid_candidates)

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry.

    Walks positions right-to-left within each row so that lowering an earlier
    position does not invalidate later positions (they are already fixed and
    their diffs are accounted for in used_diffs).
    """
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    # Precompute row lengths for speed
    row_len = k + 1

    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute diffs involving old position
                diffs_involving_old = set()
                for i in range(row_len):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < row_len:
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Search for smallest valid value
                found = None
                found_diffs = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(row_len):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        # Check against used_diffs excluding those involving old
                        if d in used_diffs and d not in diffs_involving_old:
                            valid = False
                            break
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        found = new_val
                        found_diffs = new_diffs
                        break

                if found is not None and found < old_val:
                    # Update used_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in found_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        # Squeeze pass: try to reduce the maximum entry in each row
        for r_idx in range(n_rows):
            if time.time() > deadline - 5:
                break
            row = best_rows[r_idx]
            max_pos = k  # The last position is always the max
            old_val = row[max_pos]
            
            diffs_involving_old = set()
            for i in range(row_len):
                if i == max_pos:
                    continue
                if i < max_pos:
                    diffs_involving_old.add(old_val - row[i])
                else:
                    diffs_involving_old.add(row[i] - old_val)
            
            lower_bound = row[max_pos - 1] + 1
            upper_bound = old_val - 1
            
            if upper_bound < lower_bound:
                continue
            
            found = None
            found_diffs = None
            for new_val in range(lower_bound, upper_bound + 1):
                valid = True
                new_diffs = set()
                for i in range(row_len):
                    if i == max_pos:
                        continue
                    d = new_val - row[i]
                    if d in used_diffs and d not in diffs_involving_old:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    found = new_val
                    found_diffs = new_diffs
                    break
            
            if found is not None and found < old_val:
                for d in diffs_involving_old:
                    used_diffs.discard(d)
                for d in found_diffs:
                    used_diffs.add(d)
                row[max_pos] = found
                progress = True

        if not progress:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
